#!/usr/bin/env python3
"""Bounded real-data conversion pilot; no automatic full-corpus training."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_real_conversion_pilot import bucket_for, identities, tokenizer_fingerprint
from run_single_sample_controls import fresh_destination, load_model_weights, preflight, save_checkpoint, write_json
from validate_diffusion_conversion import ByteTokenizer, preserve_rng

from llamafactory.train.diffusion.loss import (
    _get_conditioner,
    denoise_logits,
    diffusion_batch_step,
    prepare_conversion_training,
)
from llamafactory.train.diffusion.rl import sample_response


def code_hash():
    paths = [Path(__file__), ROOT / "scripts/prepare_real_conversion_pilot.py"]
    paths += [ROOT / "scripts" / p for p in ("run_single_sample_controls.py", "validate_diffusion_conversion.py")]
    paths += [ROOT / "src/llamafactory/train/diffusion" / p for p in ("loss.py", "precision.py", "rl.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["C"], manifest["K"], manifest["prompt_cap"]) != (256, 18, 4096):
        raise ValueError("Expected untruncated C=256 K=18 prompt<=4096 pilot data")
    splits, used = {}, {k: set() for k in ("prompt_hash", "question_hash", "response_hash")}
    for split in ("valid", "train"):
        path = directory / f"{split}.jsonl"
        content = path.read_bytes()
        info = manifest["splits"][split]
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError(f"Data hash mismatch: {split}")
        rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
        if len(rows) != info["rows"] or not rows:
            raise ValueError(f"Data row count mismatch: {split}")
        for row in rows:
            hashes = identities(row)
            if any(h in used[k] or row.get(k) != h for k, h in hashes.items()):
                raise ValueError("Duplicate or invalid prompt/question/response identity")
            p, r = row["prompt_ids"], row["response_ids"]
            if not 0 < len(p) <= 4096 or bucket_for(len(r)) is None or bucket_for(len(r)) != row["bucket"]:
                raise ValueError("Token length/bucket mismatch")
            if not row["prompt"].endswith("<|turn>model\n") or not row["response"].endswith("<turn|>"):
                raise ValueError("Missing turn boundary")
            for key, value in hashes.items():
                used[key].add(value)
        actual = dict(Counter(str(r["bucket"]) for r in rows))
        if actual != info["bucket_counts"] or set(actual) != {"0", "1", "2", "3"}:
            raise ValueError(f"Data bucket counts mismatch: {split}")
        splits[split] = rows
    return splits, manifest


def verify_tokens(splits, tokenizer):
    for rows in splits.values():
        for row in rows:
            for text, ids in ((row["prompt"], row["prompt_ids"]), (row["response"], row["response_ids"])):
                if any(type(i) is not int or i < 0 or i >= len(tokenizer) for i in ids):
                    raise ValueError("Token id outside tokenizer vocabulary")
                if tokenizer.encode(text, add_special_tokens=False) != ids:
                    raise ValueError("Pretokenized IDs differ from loaded tokenizer; no model loaded")
    if tokenizer.encode("<turn|>", add_special_tokens=False) != [106]:
        raise ValueError("Expected Gemma end-of-turn id 106")
    if tokenizer.eos_token_id != 1 or tokenizer.pad_token_id != 0:
        raise ValueError("Expected Gemma EOS=1, PAD=0")


def fixed_plan(splits, canvas):
    plan = {"recovery": [], "generation": []}
    for split, rows in splits.items():
        selected = []
        for bucket in range(4):
            indexes = [i for i, r in enumerate(rows) if r["bucket"] == bucket]
            if split == "valid":
                selected.extend(indexes)
            else:
                subset = indexes[:4]
                longest_index = max(indexes, key=lambda i: len(rows[i]["response_ids"]))
                if longest_index not in subset:
                    subset[-1] = longest_index
                selected.extend(subset)
            if split == "valid":
                plan["generation"].append({"split": split, "row": indexes[0], "seed": 100})
        longest = max(selected, key=lambda i: len(rows[i]["response_ids"]))
        for offset, index in enumerate(selected):
            blocks = (len(rows[index]["response_ids"]) + canvas - 1) // canvas
            block = (0, (blocks - 1) // 2, blocks - 1)[offset % 3]
            if index == longest:
                block = blocks - 1
            plan["recovery"].append({"split": split, "row": index, "block": block})
    return plan


def tensor_row(row, device):
    return tuple(torch.tensor(row[key], dtype=torch.long, device=device) for key in ("prompt_ids", "response_ids"))


def sample_pilot_response(model, prompt, settings):
    started = time.perf_counter()
    chunks, history = [], prompt
    reason = "canvas_budget"
    for block in range(settings["generation_canvases"]):
        if chunks and time.perf_counter() - started >= settings.get("generation_seconds", 120):
            reason = "time_budget"
            break
        chunk = sample_response(
            model,
            history,
            steps=settings["sample_steps"],
            canvas_size=settings["canvas"],
            max_canvases=1,
            vocab_size=model.config.text_config.vocab_size,
            sampler=settings.get("sampler", "ddim"),
            temperature=settings.get("sample_temperature", 0.7),
        )
        chunks.append(chunk)
        print(
            "GENERATION_BLOCK",
            block + 1,
            "tokens",
            sum(len(c) for c in chunks),
            "seconds",
            round(time.perf_counter() - started, 2),
            flush=True,
        )
        if len(chunk) and int(chunk[-1]) in (1, 106):
            reason = "stop_token"
            break
        history = torch.cat([history, chunk])
    return torch.cat(chunks), reason


def score_sample(row, output, tokenizer, canvas, max_canvases, stop_reason=None):
    ids = output.tolist()
    stopped = bool(ids and ids[-1] in (1, 106))
    reason = stop_reason or ("stop_token" if stopped else "canvas_budget")
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    answer = raw.split("<turn|>")[0].split("<eos>")[0]
    reference = row["response"].removesuffix("<turn|>")
    return {
        "prompt": row["prompt"],
        "reference": row["response"],
        "text": raw,
        "output_ids": ids,
        "output_tokens": len(ids),
        "target_tokens": len(row["response_ids"]),
        "generated_canvases": (len(ids) + canvas - 1) // canvas,
        "stop_reason": reason,
        "has_stop": stopped,
        "hit_canvas_cap": reason == "canvas_budget",
        "hit_time_budget": reason == "time_budget",
        "reference_exceeds_generation_cap": len(row["response_ids"]) > canvas * max_canvases,
        "reference_exact": answer.strip() == reference.strip(),
        "quality": "manual review required; reference exactness is not an open-answer quality gate",
        "has_final_channel": "<|channel>final" in raw,
    }


def aggregate_recovery(records):
    summary = []
    for split in sorted({r["split"] for r in records}):
        for noise in (0.1, 0.5, 1.0):
            group = [r for r in records if r["split"] == split and r["noise"] == noise]
            item = {"split": split, "noise": noise, "selected_canvases": len(group)}
            for region in ("body", "tail"):
                count = sum(r[f"{region}_count"] for r in group)
                changed = sum(r[f"{region}_changed"] for r in group)
                kept = sum(r[f"{region}_kept"] for r in group)
                item[region] = {
                    "tokens": count,
                    "ce": sum(r[f"{region}_ce_sum"] for r in group) / count if count else None,
                    "restored": sum(r[f"{region}_restored"] for r in group) / changed if changed else None,
                    "preserved": sum(r[f"{region}_preserved"] for r in group) / kept if kept else None,
                }
            summary.append(item)
    return summary


@torch.inference_mode()
def evaluate(model, tokenizer, splits, plan, settings, out, step, generate):
    was_training = model.training
    model.eval()
    canvas = settings["canvas"]
    device = next(model.parameters()).device
    vocab = model.config.text_config.vocab_size
    report = {"step": step, "recovery": [], "samples": [], "generation_requested": generate}
    try:
        for item in plan["recovery"]:
            row = splits[item["split"]][item["row"]]
            prompt, target = tensor_row(row, device)
            begin = item["block"] * canvas
            body = target[begin : begin + canvas]
            clean = target.new_full((canvas,), 1)
            clean[: len(body)] = body
            history = torch.cat([prompt, target[:begin]])
            for noise in (0.1, 0.5, 1.0):
                torch.manual_seed(90000 + item["row"] * 100 + item["block"] * 10 + int(noise * 10))
                changed = torch.rand(canvas, device=device) < noise
                noisy = torch.where(changed, torch.randint(0, vocab, (canvas,), device=device), clean)
                logits = denoise_logits(model, history, noisy, pad_id=0)[0]
                ce = F.cross_entropy(logits.float(), clean, reduction="none")
                correct = logits.argmax(-1) == clean
                record = dict(item, bucket=row["bucket"], noise=noise, context="gold_previous_canvases")
                for region, region_slice in (("body", slice(None, len(body))), ("tail", slice(len(body), None))):
                    moved, right = changed[region_slice], correct[region_slice]
                    record.update(
                        {
                            f"{region}_count": len(right),
                            f"{region}_changed": int(moved.sum()),
                            f"{region}_restored": int((moved & right).sum()),
                            f"{region}_kept": int((~moved).sum()),
                            f"{region}_preserved": int((~moved & right).sum()),
                            f"{region}_ce_sum": float(ce[region_slice].sum()),
                        }
                    )
                report["recovery"].append(record)
            if (len(report["recovery"]) // 3) % 16 == 0:
                print("RECOVERY_PROGRESS", step, len(report["recovery"]) // 3, "/", len(plan["recovery"]), flush=True)
        if generate:
            for item in plan["generation"]:
                row = splits[item["split"]][item["row"]]
                prompt = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                torch.manual_seed(item["seed"])
                print("GENERATE_START", step, item, "max_canvases", settings["generation_canvases"], flush=True)
                output, reason = sample_pilot_response(model, prompt, settings)
                record = dict(
                    item, **score_sample(row, output, tokenizer, canvas, settings["generation_canvases"], reason)
                )
                report["samples"].append(record)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(dict(step=step, **record), ensure_ascii=False) + "\n")
                print("ANSWER", step, item["row"], record["stop_reason"], repr(record["text"][:240]), flush=True)
        report["aggregate"] = aggregate_recovery(report["recovery"])
        write_json(out / f"eval-{step:04d}.json", report)
        for item in report["aggregate"]:
            print("RECOVERY", step, json.dumps(item), flush=True)
        return report
    finally:
        model.train(was_training)


def schedule_index(micro_step, size, seed=42):
    epoch, offset = divmod(micro_step, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def tiny_setup():
    tokenizer = ByteTokenizer()
    config = Gemma4TextConfig(
        vocab_size=259,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_size_per_layer_input=4,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=config)).to(torch.bfloat16)
    splits = {}
    for split in ("train", "valid"):
        splits[split] = [
            {
                "prompt": f"{split}{i}",
                "response": "x" * (i * 16 + 1),
                "bucket": i,
                "prompt_ids": tokenizer.encode(f"{split}{i}"),
                "response_ids": tokenizer.encode("x" * (i * 16 + 1)) + [1],
            }
            for i in range(4)
        ]
    return model, tokenizer, splits


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_conversion_pilot_v1")
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/e2b_conversion_pilot_v3"))
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--generation-every", type=int, default=256)
    parser.add_argument("--generation-canvases", type=int, default=18)
    parser.add_argument(
        "--generation-seconds",
        type=int,
        default=120,
        help="Per-answer soft time limit, checked between canvases; always attempts one full canvas",
    )
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--sampler", choices=["ddim", "entropy"], default="ddim")
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--optimizer-mode", choices=["fp32-master", "bf16-adamw"], default="fp32-master")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume", type=Path, help="Trusted complete pilot checkpoint; use a fresh --out")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.max_steps,
            args.grad_accum,
            args.eval_every,
            args.generation_every,
            args.sample_steps,
            args.generation_seconds,
            args.keep_checkpoints,
        )
        < 1
    ):
        parser.error("All budgets must be positive")
    if args.sample_temperature <= 0:
        parser.error("sample-temperature must be positive")
    if not 1 <= args.generation_canvases <= 18:
        parser.error("Generation canvas budget must be 1..18")
    if args.generation_every % args.eval_every:
        parser.error("generation-every must be a multiple of eval-every")
    return args


def main():
    args = arguments()
    if args.summary_only:
        print((args.out / "summary.json").read_text(), flush=True)
        for path in sorted(args.out.glob("eval-*.json")):
            report = json.loads(path.read_text())
            print("EVAL_SUMMARY", report["step"], json.dumps(report["aggregate"]), flush=True)
        return
    fresh_destination(args.out)
    splits, manifest = ({}, {}) if args.tiny else load_data(args.data)
    hardware = preflight(args.tiny, args.out)
    print("PREFLIGHT", hardware, flush=True)
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("out", "resume", "summary_only", "data", "keep_checkpoints")
    }
    settings.update(
        canvas=16 if args.tiny else 256,
        capacity=18,
        lr=1.125e-4,
        warmup=50,
        optimizer=(
            "MasterAdamW (CPU FP32 master + moments)"
            if args.optimizer_mode == "fp32-master"
            else "BF16 AdamW; foreach=False"
        ),
        betas=[0.95, 0.99],
        weight_decay=1e-4,
        loss_mode="canvas-mean",
        rl_steps=0,
        code_sha256=code_hash(),
        data_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        scope="experimental section-4 conversion pilot; optimizer/budget are not a published Google recipe",
        recovery_scope="one fixed selected canvas per valid row and up to four train rows per length bucket; gold history",
        generation_scope="one fixed heldout prompt per length bucket, seed 100; prompt-only free generation",
        generation_time_policy="soft limit checked between canvases; one full canvas always attempted; caps are not quality failures",
    )
    restored = None
    if args.resume:
        if not (args.resume / "COMPLETE").is_file():
            raise RuntimeError("Incomplete checkpoint")
        restored = torch.load(args.resume / "diagnostic_state.pt", map_location="cpu", weights_only=False)
        if restored["settings"] != settings:
            raise RuntimeError("Checkpoint settings/code/data differ; no model loaded")
        if restored["global_step"] >= args.max_steps:
            raise ValueError("Checkpoint already reached the configured budget")
    random.seed(42)
    torch.manual_seed(42)
    if args.tiny:
        model, tokenizer, splits = tiny_setup()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        fingerprint = tokenizer_fingerprint(tokenizer)
        if any(
            fingerprint[key] != manifest["tokenizer"][key]
            for key in ("sha256", "fingerprint_method", "vocab_size", "length")
        ):
            raise ValueError("Tokenizer fingerprint differs from data manifest; no model loaded")
        verify_tokens(splits, tokenizer)
        model, info = Gemma4ForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            output_loading_info=True,
        )
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Original model loading mismatch: {info}")
        model = model.cuda()
    prepare_conversion_training(model)
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        raise RuntimeError("Embedding/lm_head weights must remain tied")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.optimizer_mode == "fp32-master":
        from llamafactory.train.diffusion.precision import MasterAdamW, require_master_memory

        if not args.tiny:
            print("MASTER_MEMORY", require_master_memory(model), flush=True)
        optimizer = MasterAdamW(
            model.parameters(), lr=settings["lr"], betas=(0.95, 0.99), weight_decay=settings["weight_decay"]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["lr"],
            betas=(0.95, 0.99),
            weight_decay=settings["weight_decay"],
            foreach=False,
        )
    step = 0
    if restored:
        load_model_weights(model, args.resume)
        _get_conditioner(model).load_state_dict(
            load_file(str(args.resume / "self_conditioner.safetensors")), strict=True
        )
        optimizer.load_state_dict(restored["optimizer"])
        step = restored["global_step"]
        torch.set_rng_state(restored["cpu_rng"])
        if restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        random.setstate(restored["python_rng"])
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "settings.json", settings)
    write_json(args.out / "data_manifest.json", manifest)
    plan = fixed_plan(splits, settings["canvas"])
    write_json(args.out / "eval_plan.json", plan)
    summary = {
        "status": "running",
        "global_step": step,
        "max_steps": args.max_steps,
        "quality": "not automatically assessed; inspect unseen-prompt samples",
        "resume_from": str(args.resume) if args.resume else None,
    }
    write_json(args.out / "summary.json", summary)
    print(
        "BUDGET",
        {
            "optimizer_steps": args.max_steps,
            "rows_per_step": args.grad_accum,
            "train_rows": len(splits["train"]),
            "valid_rows": len(splits["valid"]),
            "generation_prompts": len(plan["generation"]),
            "generation_canvases": args.generation_canvases,
            "sample_steps": args.sample_steps,
            "generation_soft_seconds_per_answer": args.generation_seconds,
        },
        flush=True,
    )
    if not restored:
        with preserve_rng():
            evaluate(model, tokenizer, splits, plan, settings, args.out, 0, True)
    device = next(model.parameters()).device
    block_counts = Counter(restored.get("sampled_block_counts", {}) if restored else {})
    with (args.out / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < args.max_steps:
            started = time.perf_counter()
            step += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = settings["lr"] * min(step / settings["warmup"], 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            records, loss_sum = [], 0.0
            for accumulation in range(args.grad_accum):
                micro = (step - 1) * args.grad_accum + accumulation
                index = schedule_index(micro, len(splits["train"]))
                prompt, target = tensor_row(splits["train"][index], device)
                torch.manual_seed(420000 + micro)
                loss, parts = diffusion_batch_step(
                    model,
                    [prompt],
                    [target],
                    canvas_size=settings["canvas"],
                    num_canvases=18,
                    pad_id=0,
                    eos_id=1,
                    diagnostics=True,
                    loss_mode="canvas-mean",
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; stopped without advancing to full data")
                (loss / args.grad_accum).backward()
                loss_sum += float(loss.detach()) / args.grad_accum
                records.append(dict(row=index, **parts["rows"][0]))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient; stopped")
            if step == 1:
                print(
                    "PARAMETERS_WITH_GRAD",
                    sum(p.numel() for p in model.parameters() if p.grad is not None),
                    flush=True,
                )
            optimizer.step()
            block_counts.update(str(r["selected_block"]) for r in records)
            record = {
                "step": step,
                "loss": loss_sum,
                "grad": float(norm),
                "lr": lr,
                "seconds": time.perf_counter() - started,
                "rows": records,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print("TRAIN", json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                optimizer.zero_grad(set_to_none=True)
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    args.out,
                    {
                        "global_step": step,
                        "settings": settings,
                        "sampled_block_counts": dict(block_counts),
                    },
                    keep=args.keep_checkpoints,
                )
                with preserve_rng():
                    evaluate(
                        model,
                        tokenizer,
                        splits,
                        plan,
                        settings,
                        args.out,
                        step,
                        step % args.generation_every == 0 or step == args.max_steps,
                    )
                summary.update(
                    global_step=step,
                    sampled_block_counts=dict(block_counts),
                    highest_training_block_1based=1 + max(map(int, block_counts)),
                )
                write_json(args.out / "summary.json", summary)
    summary.update(status="budget_complete_no_full_training", global_step=step)
    write_json(args.out / "summary.json", summary)
    print("REAL_PILOT_DONE", args.out, "No full-data training started.", flush=True)


if __name__ == "__main__":
    main()
