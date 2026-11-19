#!/usr/bin/env python3
"""Frozen pilot checkpoint: first-canvas and gold-history trajectory diagnostics."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_length_seeds import preflight
from run_real_conversion_pilot import code_hash, load_data, tensor_row, verify_tokens
from run_single_sample_controls import fresh_destination, write_json
from validate_diffusion_conversion import ByteTokenizer, preserve_rng

from llamafactory.train.diffusion.loss import (
    _embed_distribution,
    _get_conditioner,
    _token_embedding,
    denoise_logits,
    denoise_step_logits,
)
from llamafactory.train.diffusion.rl import _commit, _temperature


def checkpoint_inventory(checkpoint):
    return {
        str(p.relative_to(checkpoint)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in checkpoint.rglob("*")
        if p.is_file()
    }


def validate(checkpoint, data, out, tiny):
    fresh_destination(out)
    if out.resolve() == checkpoint.resolve() or out.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("Audit output must be outside the checkpoint")
    if checkpoint.resolve().is_relative_to(out.resolve()):
        raise ValueError("Audit output cannot contain the checkpoint")
    if not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError("Incomplete checkpoint; no model loaded")
    settings = json.loads((checkpoint.parent / "settings.json").read_text())
    if bool(settings.get("tiny")) != tiny or settings.get("code_sha256") != code_hash():
        raise RuntimeError("Checkpoint mode/training source mismatch; use the exact v4 runtime")
    if settings.get("canvas") != (16 if tiny else 256) or settings.get("capacity") != 18:
        raise RuntimeError("Unexpected canvas/capacity settings")
    if not tiny and settings.get("sample_steps") != 48:
        raise RuntimeError("Expected the saved maximum 48-step pilot")
    if not tiny and settings.get("model") == str(checkpoint):
        raise RuntimeError("Pilot settings do not identify original model initialization")
    complete_step = int((checkpoint / "COMPLETE").read_text())
    if complete_step != 512 and not tiny:
        raise ValueError("This audit targets completed checkpoint-512")
    for name in ("config.json", "self_conditioner.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not any((checkpoint / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
        raise FileNotFoundError("Checkpoint has no model weights")
    if tiny:
        splits, manifest = {}, {}
    else:
        splits, manifest = load_data(data)
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if fingerprint != settings.get("data_sha256"):
            raise RuntimeError("Pilot data differ from saved training data")
        if manifest != json.loads((checkpoint.parent / "data_manifest.json").read_text()):
            raise RuntimeError("Saved pilot manifest differs from packaged data")
    return settings, splits, manifest


def audit_plan(splits, canvas):
    plan = []
    for split in ("train", "valid"):
        rows = splits[split]
        for bucket in range(4):
            index = next(i for i, row in enumerate(rows) if row["bucket"] == bucket)
            plan.append({"split": split, "row": index, "bucket": bucket, "block": 0, "context": "prompt_only"})
        longest = max(range(len(rows)), key=lambda i: len(rows[i]["response_ids"]))
        last = (len(rows[longest]["response_ids"]) - 1) // canvas
        if last:
            plan.append(
                {
                    "split": split,
                    "row": longest,
                    "bucket": rows[longest]["bucket"],
                    "block": last,
                    "context": "gold_previous_canvases",
                }
            )
    return plan


def first_stop(tokens):
    hits = ((tokens == 1) | (tokens == 106)).nonzero().flatten()
    return int(hits[0]) if len(hits) else None


def score(logits, sample, target, body_tokens):
    ce = F.cross_entropy(logits[0].float(), target, reduction="none")
    pred = logits[0].argmax(-1)
    result = {}
    for name, selection in (("body", slice(None, body_tokens)), ("tail", slice(body_tokens, None))):
        expected = target[selection]
        n = len(expected)
        result[name] = {
            "tokens": n,
            "ce": float(ce[selection].mean()) if n else None,
            "argmax_reference_accuracy": float((pred[selection] == expected).float().mean()) if n else None,
            "sample_reference_accuracy": float((sample[selection] == expected).float().mean()) if n else None,
        }
    stop = first_stop(sample)
    expected_stop = first_stop(target)
    visible = sample if stop is None else sample[: stop + 1]
    counts = Counter(visible.tolist())
    result.update(
        sample_stop_index=stop,
        argmax_stop_index=first_stop(pred),
        reference_stop_index=expected_stop,
        stop_before_reference=stop is not None and (expected_stop is None or stop < expected_stop),
        most_frequent_token_fraction=max(counts.values()) / len(visible),
        unique_tokens=len(counts),
    )
    return result


@torch.inference_mode()
def trajectory(model, history, target, body_tokens, *, steps, seed, self_conditioning, seconds):
    config = model.config.text_config
    embedding = _token_embedding(model)
    torch.manual_seed(seed)
    canvas = torch.randint(0, config.vocab_size, (1, len(target)), device=history.device)
    initial = canvas.clone()
    zero = torch.zeros(1, len(target), config.hidden_size, dtype=embedding.weight.dtype, device=history.device)
    signal = zero
    records, reason = [], "step_budget"
    started = time.perf_counter()
    for step in range(steps):
        if records and time.perf_counter() - started >= seconds:
            reason = "time_budget"
            break
        logits = denoise_step_logits(model, history, canvas[0], sc_embeddings=signal, pad_id=0)
        nxt, shaped, entropy = _commit(
            logits,
            canvas,
            temperature=_temperature(step, steps),
            entropy_bound=0.1,
            vocab_size=config.vocab_size,
            is_final_step=step + 1 == steps,
        )
        stable = bool((shaped.argmax(-1) == canvas).all())
        confident = bool(entropy.mean() <= 0.005)
        record = dict(
            step=step + 1,
            temperature=_temperature(step, steps),
            entropy=float(entropy.mean()),
            stable=stable,
            confident=confident,
            changed_tokens=int((nxt != canvas).sum()),
            input_ids=canvas[0].tolist(),
            sampled_ids=nxt[0].tolist(),
            argmax_ids=logits[0].argmax(-1).tolist(),
            **score(logits, nxt[0], target, body_tokens),
        )
        records.append(record)
        signal = _embed_distribution(embedding, shaped) if self_conditioning else zero
        canvas = nxt
        if stable and confident:
            reason = "entropy_and_stability"
            break
    return {
        "self_conditioning": "carried" if self_conditioning else "zero_each_step",
        "initial_ids": initial[0].tolist(),
        "output_ids": canvas[0].tolist(),
        "used_steps": len(records),
        "stop_reason": reason,
        "seconds": time.perf_counter() - started,
        "trace": records,
    }


@torch.inference_mode()
def execute(model, tokenizer, splits, plan, out, steps, seed, seconds, canvas_size):
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    records = []
    for item in plan:
        row = splits[item["split"]][item["row"]]
        prompt, response = tensor_row(row, device)
        begin = item["block"] * canvas_size
        body = response[begin : begin + canvas_size]
        target = response.new_full((canvas_size,), 1)
        target[: len(body)] = body
        history = torch.cat([prompt, response[:begin]])
        torch.manual_seed(seed)
        initial = torch.randint(0, model.config.text_config.vocab_size, (canvas_size,), device=device)
        two_pass = denoise_logits(model, history, initial, pad_id=0)
        predictor = score(two_pass, two_pass[0].argmax(-1), target, len(body))
        for carry in (True, False):
            with preserve_rng():
                result = trajectory(
                    model, history, target, len(body), steps=steps, seed=seed, self_conditioning=carry, seconds=seconds
                )
            if result["initial_ids"] != initial.tolist():
                raise RuntimeError("Trajectory and two-pass reference used different initial noise")
            output = torch.tensor(result["output_ids"], dtype=torch.long)
            stop = first_stop(output)
            record = dict(
                item,
                seed=seed,
                prompt=row["prompt"],
                reference=row["response"],
                target_canvas_ids=target.tolist(),
                body_tokens=len(body),
                initial_two_pass_predictor=predictor,
                **result,
            )
            record["text"] = tokenizer.decode(output.tolist(), skip_special_tokens=False)
            record["visible_text"] = tokenizer.decode(
                output.tolist() if stop is None else output[: stop + 1].tolist(), skip_special_tokens=False
            )
            record["snapshot_texts"] = [
                {"step": r["step"], "text": tokenizer.decode(r["sampled_ids"], skip_special_tokens=False)}
                for r in result["trace"]
                if r["step"] in (1, 8, 24, steps) or r is result["trace"][-1]
            ]
            with (out / "trajectories.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            final = result["trace"][-1]
            print(
                "TRAJECTORY",
                item,
                result["self_conditioning"],
                "used",
                result["used_steps"],
                result["stop_reason"],
                "body_CE",
                round(final["body"]["ce"], 4),
                "stop",
                final["sample_stop_index"],
                repr(record["visible_text"][:120]),
                flush=True,
            )
    return records


def summarize(records):
    if not records or len(records) % 2:
        raise ValueError("Expected complete pairs of SC audit records")
    comparisons = []
    for on, off in zip(records[::2], records[1::2]):
        if on["self_conditioning"] != "carried" or off["self_conditioning"] != "zero_each_step":
            raise ValueError("Invalid SC audit control ordering")
        if (on["split"], on["row"], on["block"], on["seed"]) != (off["split"], off["row"], off["block"], off["seed"]):
            raise ValueError("Unpaired SC audit records")
        if on["initial_ids"] != off["initial_ids"]:
            raise ValueError("SC audit initial noise differs")
        comparisons.append({key: on[key] for key in ("split", "row", "block", "context")})
        for mode, record in (("carried", on), ("zero_each_step", off)):
            comparisons[-1][mode] = {
                "used_steps": record["used_steps"],
                "stop_reason": record["stop_reason"],
                "initial": record["trace"][0]["body"],
                "initial_two_pass_predictor": record["initial_two_pass_predictor"],
                "final": record["trace"][-1]["body"],
                "sample_stop_index": record["trace"][-1]["sample_stop_index"],
                "most_frequent_token_fraction": record["trace"][-1]["most_frequent_token_fraction"],
            }
    return {
        "scope": "Frozen checkpoint, selected canvases, one seed; reference scores are diagnostic, not open-answer quality or full-response generalization gates",
        "trajectories": len(records),
        "comparisons": comparisons,
        "quality": "Inspect raw text and trajectories; no automatic quality pass",
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/content/drive/MyDrive/saves/e2b_conversion_compare_v4/checkpoint-512"),
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_conversion_pilot_v1")
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/e2b_frozen_trajectory_v1"))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seconds-per-trajectory", type=int, default=120)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        print((args.out / "summary.json").read_text(), flush=True)
        return
    if args.seed < 0 or args.seconds_per_trajectory < 1:
        parser.error("Use a nonnegative seed and positive soft time budget")
    settings, splits, manifest = validate(args.checkpoint, args.data, args.out, args.tiny)
    hardware = preflight(args.tiny)
    before = checkpoint_inventory(args.checkpoint)
    tokenizer = ByteTokenizer() if args.tiny else AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if not args.tiny:
        verify_tokens(splits, tokenizer)
    model, info = Gemma4ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, local_files_only=True, output_loading_info=True
    )
    unexpected = [name for name in info.get("unexpected_keys", []) if not name.startswith("self_conditioner.")]
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")) or unexpected:
        raise RuntimeError(f"Checkpoint model load mismatch: {info}")
    _get_conditioner(model).load_state_dict(
        load_file(str(args.checkpoint / "self_conditioner.safetensors")), strict=True
    )
    model = model.to("cpu" if args.tiny else "cuda").eval().requires_grad_(False)
    if args.tiny:
        splits = {
            split: [
                {
                    "prompt": f"{split}{i}",
                    "response": "x" * (i * 16 + 1),
                    "bucket": i,
                    "prompt_ids": tokenizer.encode(f"{split}{i}"),
                    "response_ids": tokenizer.encode("x" * (i * 16 + 1)) + [1],
                }
                for i in range(4)
            ]
            for split in ("train", "valid")
        }
    plan = audit_plan(splits, settings["canvas"])
    steps = settings["sample_steps"]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "settings.json",
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": 512 if not args.tiny else int((args.checkpoint / "COMPLETE").read_text()),
            "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_code_sha256": settings["code_sha256"],
            "seed": args.seed,
            "canvas": settings["canvas"],
            "max_steps": steps,
            "max_trajectories": 2 * len(plan),
            "max_forward_calls": (2 + 2 * steps) * len(plan),
            "plan": plan,
            "hardware": hardware,
            "actual_attention_backend": model.model.language_model.config._attn_implementation,
            "controls": "carried SC follows production sampler; zero_each_step is a nonofficial diagnostic ablation",
            "metric_semantics": "CE and argmax score raw denoiser logits on each input canvas; sampled IDs intentionally include renoised positions until the final step. Reference matches are not open-answer quality judgments.",
            "time_policy": "soft limit checked between denoising steps",
            "data_manifest": manifest,
            "checkpoint_inventory_before": before,
            "optimizer_created_or_loaded": False,
        },
    )
    print("FROZEN_AUDIT_BUDGET", len(plan), "canvases", 2 * len(plan), "trajectories", "max_steps", steps, flush=True)
    records = execute(
        model, tokenizer, splits, plan, args.out, steps, args.seed, args.seconds_per_trajectory, settings["canvas"]
    )
    if before != checkpoint_inventory(args.checkpoint):
        raise RuntimeError("Checkpoint file sizes or modification times changed during audit")
    summary = summarize(records)
    summary["checkpoint_inventory_unchanged"] = True
    write_json(args.out / "summary.json", summary)
    print("FROZEN_TRAJECTORY_AUDIT_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
