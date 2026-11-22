#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Bounded 1 -> 2 -> 4 arithmetic diagnostic; never starts full-data training."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validate_diffusion_conversion import ByteTokenizer, encode, preserve_rng

from llamafactory.train.diffusion.loss import _get_conditioner, denoise_logits, diffusion_batch_step, prepare_conversion_training
from llamafactory.train.diffusion.rl import sample_canvas, sample_response


def task_rows(suite="arithmetic"):
    if suite == "length":
        short = {
            "kind": "short_text",
            "prompt": "用三句话说明午休后整理桌面的做法。",
            "response": "先把水杯放到稳妥的位置，再收起桌面上不用的纸张。用干净的布擦去灰尘，把常用物品摆在顺手的位置。最后留出一块空白区域，方便下午开始工作。",
        }
        long = {
            "kind": "cross_block_text",
            "prompt": "完整介绍一次社区图书整理活动，包括准备、分类、登记、现场协作和收尾。",
            "response": (
                "周六上午，社区活动室迎来了一次图书整理活动。工作人员提前打开窗户，摆好长桌，并在入口处准备了签到表。参加者带来的书先放在收集区，由志愿者逐本检查书脊和封面。\n\n"
                "检查完成后，大家按照儿童读物、文学作品、生活知识和工具参考四类分组。破损严重的书单独放置，等待修补；重复的书保留清楚的登记记录，方便以后交换。分类时遇到拿不准的内容，先请另一位志愿者复核。\n\n"
                "登记组负责记录书名、作者和适合的阅读年龄，再给每本书贴上编号。书架旁留出通道，较重的书放在下层，儿童读物放在容易拿取的位置。每完成一层，整理者都会核对标签，避免放错位置。\n\n"
                "现场还有两位居民负责引导新来的参加者。他们介绍各组任务，提醒大家轻拿轻放，也把需要帮助的问题及时转交给工作人员。有人负责搬运，有人负责检查，大家可以根据体力和兴趣选择工作。\n\n"
                "临近中午，各组一起清点已经整理好的图书，收起剩余的标签和纸箱。工作人员把待修补的书放进专用柜，确认借阅登记本留在服务台。最后，大家清洁桌面、关闭窗户，并约好下次开放借阅的时间。"
            ),
        }
        heldout = [
            {
                "kind": "short_text",
                "prompt": "用三句话说明出门前检查背包的做法。",
                "response": "先确认钥匙和钱包已经放好，再检查水杯是否拧紧。把需要携带的文件装进独立夹层，避免折损。最后拉好拉链，背起包后检查重量是否合适。",
            },
        ]
        return [short, long], heldout
    if suite != "arithmetic":
        raise ValueError(f"Unknown task suite: {suite}")

    def row(a, b):
        return {"kind": "math", "prompt": f"只回答整数：{a}+{b}等于多少？", "response": str(a + b)}

    return [row(2, 3), row(3, 4), row(4, 5), row(5, 6)], [row(6, 7), row(10, 11)]


def length_metadata(tokenizer, rows, canvas):
    lengths = [len(tokenizer.encode(row["response"] + "<turn|>", add_special_tokens=False)) for row in rows]
    if not (32 <= lengths[0] <= 128 and canvas < lengths[1] <= 2 * canvas):
        raise ValueError(f"Length-suite targets must be 32..128 and 257..512 tokens with C=256; got {lengths}")
    return [
        {"row": i, "kind": row["kind"], "target_tokens": n, "blocks": (n + canvas - 1) // canvas}
        for i, (row, n) in enumerate(zip(rows, lengths))
    ]


def compare_blocks(output, target, canvas):
    results = []
    for begin in range(0, target.numel(), canvas):
        expected = target[begin : begin + canvas]
        actual = output[begin : begin + len(expected)]
        exact = actual.numel() == expected.numel() and torch.equal(actual, expected)
        correct = int((actual == expected[: actual.numel()]).sum())
        results.append(
            dict(
                block=begin // canvas,
                expected_tokens=expected.numel(),
                available_tokens=actual.numel(),
                exact=exact,
                token_accuracy=correct / expected.numel(),
            )
        )
    return results


def fresh_destination(out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise RuntimeError(
            "Output already exists. Choose a fresh empty --out, including for --resume; nothing overwritten"
        )


def preflight(tiny=False, out=None):
    if tiny:
        return {"mode": "tiny_cpu", "checks": "execution only; no E2B capability claim"}
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. No model download or training started.")
    free, total = torch.cuda.mem_get_info()
    info = {"gpu": torch.cuda.get_device_name(), "total_gib": total / 1024**3, "free_gib": free / 1024**3}
    if info["total_gib"] < 75 or info["free_gib"] < 45:
        raise RuntimeError(f"Requires an idle 80GB-class GPU, not L4/T4/40GB. Found {info}. No training started.")
    import psutil

    cpu_free = psutil.virtual_memory().available / 1024**3
    local_free = shutil.disk_usage(Path.cwd()).free / 1024**3
    destination = Path(out) if out is not None else Path.cwd()
    while not destination.exists():
        destination = destination.parent
    output_free = shutil.disk_usage(destination).free / 1024**3
    info.update(cpu_free_gib=cpu_free, local_disk_free_gib=local_free, output_disk_free_gib=output_free)
    # Conservative diagnostic limits, not measured E2B peak requirements.
    if cpu_free < 32 or local_free < 40 or output_free < 100:
        raise RuntimeError(
            f"Requires >=32 GiB free host RAM, >=40 GiB local disk and >=100 GiB checkpoint disk. "
            f"Found {info}. No model download or training started."
        )
    return info


def load_model_weights(model, checkpoint):
    from safetensors import safe_open

    index = checkpoint / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.exists() else None
    names = sorted(set(mapping.values())) if mapping is not None else ["model.safetensors"]
    expected = model.state_dict()
    observed = {}
    for name in names:
        if Path(name).name != name:
            raise RuntimeError(f"Invalid diagnostic shard filename: {name}")
        with safe_open(str(checkpoint / name), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in observed:
                    raise RuntimeError(f"Duplicate checkpoint key: {key}")
                observed[key] = name
                if key in expected and tuple(shard.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    raise RuntimeError(f"Invalid diagnostic weight shape: {key}")
    if mapping is not None and observed != mapping:
        raise RuntimeError("Diagnostic shard contents disagree with weight index")
    missing = set(expected) - set(observed)
    unexpected = set(observed) - set(expected)
    # Transformers omits one alias when embedding and LM head weights are tied.
    allowed = set()
    if model.lm_head.weight is model.get_input_embeddings().weight:
        tied = {
            name for name, param in model.named_parameters(remove_duplicate=False) if param is model.lm_head.weight
        }
        if tied - missing:
            allowed = tied
    if missing - allowed or unexpected:
        raise RuntimeError(f"Invalid diagnostic weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    with torch.no_grad():
        for name in names:
            with safe_open(str(checkpoint / name), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    expected[key].copy_(shard.get_tensor(key))


def source_hash():
    paths = [Path(__file__), ROOT / "scripts/validate_diffusion_conversion.py"]
    paths += [ROOT / "src/llamafactory/train/diffusion" / p for p in ("loss.py", "rl.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_answer(text):
    return text.split("<turn|>")[0].split("<eos>")[0].strip()


def stable_pass(report, streak):
    attempts = [r["passed"] for r in report["samples"] if r["split"] == "train"]
    return streak + 1 if attempts and all(attempts) else 0


@torch.inference_mode()
def evaluate(model, tokenizer, active, heldout, args, canvas):
    was_training = model.training
    model.eval()
    is_length = getattr(args, "suite", "arithmetic") == "length"
    report = {
        "samples": [],
        "recovery": [],
        "teacher_blocks": [],
        "sampling": {"steps": args.sample_steps, "max_canvases": 3 if is_length else 1},
    }
    device = next(model.parameters()).device
    vocab = model.config.text_config.vocab_size
    try:
        for split, rows in (("train", active), ("heldout", heldout)):
            for index, row in enumerate(rows):
                prompt, target = encode(tokenizer, row, device, canvas * 18)
                if target.numel() > (2 * canvas if is_length else canvas):
                    raise ValueError("Diagnostic target exceeds configured one/two-canvas length")
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    output = sample_response(
                        model,
                        prompt,
                        steps=args.sample_steps,
                        canvas_size=canvas,
                        max_canvases=3 if is_length else 1,
                        vocab_size=vocab,
                    )
                    text = tokenizer.decode(output.tolist(), skip_special_tokens=False)
                    report["samples"].append(
                        {
                            "split": split,
                            "row": index,
                            "prompt": row["prompt"],
                            "seed": seed,
                            "expected": row["response"],
                            "text": text,
                            "passed": clean_answer(text) == row["response"],
                            "output_tokens": output.numel(),
                            "has_stop": bool(((output == 1) | (output == 106)).any()),
                            "target_tokens": target.numel(),
                            "blocks": compare_blocks(output, target, canvas),
                        }
                    )
                for begin in range(0, target.numel(), canvas):
                    body = target[begin : begin + canvas]
                    history = torch.cat([prompt, target[:begin]])
                    clean = target.new_full((canvas,), 1)
                    clean[: len(body)] = body
                    for fraction in (0.1, 0.5, 1.0):
                        torch.manual_seed(9000 + index * 100 + begin + int(fraction * 10))
                        changed = torch.rand(clean.shape, device=device) < fraction
                        noisy = torch.where(changed, torch.randint(0, vocab, clean.shape, device=device), clean)
                        logits = denoise_logits(model, history, noisy, pad_id=0)[0]
                        ce = F.cross_entropy(logits.float(), clean, reduction="none")
                        pred = logits.argmax(-1)
                        report["recovery"].append(
                            {
                                "split": split,
                                "row": index,
                                "block": begin // canvas,
                                "noise": fraction,
                                "body_count": len(body),
                                "body_changed": int(changed[: len(body)].sum()),
                                "body_restored": int(((pred == clean) & changed)[: len(body)].sum()),
                                "body_ce_sum": float(ce[: len(body)].sum()),
                                "tail_count": canvas - len(body),
                                "tail_ce_sum": float(ce[len(body) :].sum()),
                                "tail_changed": int(changed[len(body) :].sum()),
                                "tail_restored": int(((pred == clean) & changed)[len(body) :].sum()),
                                "tail_kept": int((~changed)[len(body) :].sum()),
                                "tail_preserved": int(((pred == clean) & ~changed)[len(body) :].sum()),
                            }
                        )
                    if is_length and split == "train" and target.numel() > canvas:
                        for seed in args.seeds:
                            torch.manual_seed(seed)
                            predicted, _ = sample_canvas(
                                model, history, steps=args.sample_steps, canvas_size=canvas, vocab_size=vocab
                            )
                            report["teacher_blocks"].append(
                                {
                                    "split": split,
                                    "row": index,
                                    "block": begin // canvas,
                                    "seed": seed,
                                    "context": "correct_prefix",
                                    "result": compare_blocks(predicted[0], body, canvas)[0],
                                    "text": tokenizer.decode(predicted[0].tolist(), skip_special_tokens=False),
                                }
                            )
        return report
    finally:
        model.train(was_training)


def save_checkpoint(model, tokenizer, optimizer, out, state, *, keep: int = 2, save_optimizer: bool = True):
    from safetensors.torch import save_file

    if keep < 1:
        raise ValueError("keep must be >= 1")
    step = state["global_step"]
    staging = out / f".checkpoint-{step}.writing"
    target = out / f"checkpoint-{step}"
    if staging.exists() or target.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {target}. Nothing overwritten.")
    staging.mkdir()
    model.save_pretrained(staging)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(staging)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in _get_conditioner(model).state_dict().items()},
        str(staging / "self_conditioner.safetensors"),
    )
    state = dict(
        state,
        optimizer=optimizer.state_dict() if save_optimizer else None,
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        python_rng=random.getstate(),
    )
    torch.save(state, staging / "diagnostic_state.pt")
    (staging / "COMPLETE").write_text(str(step))
    staging.rename(target)
    write_json(out / "latest_checkpoint.json", {"path": str(target)})
    print("CHECKPOINT_COMPLETE", target, flush=True)
    # Preserve `keep` complete recovery points; never delete an unfinished save.
    checkpoints = sorted(
        [p for p in out.glob("checkpoint-*") if (p / "COMPLETE").is_file()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for previous in checkpoints[:-keep]:
        shutil.rmtree(previous)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--out", default="/content/drive/MyDrive/saves/e2b_single_sample_v1")
    parser.add_argument("--suite", choices=["arithmetic", "length"], default="arithmetic")
    parser.add_argument("--stages", type=int, nargs="+", default=None)
    parser.add_argument("--steps-per-stage", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="Local trusted checkpoint created by this script")
    args = parser.parse_args()
    if args.stages is None:
        args.stages = [1, 2] if args.suite == "length" else [1, 2, 4]
    if args.suite == "length" and any(n not in (1, 2) for n in args.stages):
        parser.error("Length suite has only stages 1 and 2")
    if (
        args.steps_per_stage < 1
        or args.eval_every < 1
        or args.sample_steps < 1
        or args.stages != sorted(set(args.stages))
        or any(n not in (1, 2, 4) for n in args.stages)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Positive budgets, unique seeds, and ascending stages chosen from 1,2,4 are required")
    if not args.tiny and args.steps_per_stage < 2 * args.eval_every:
        parser.error("Budget must allow two successive evaluations for the stability gate")
    return args


def print_report(report):
    for split in ("train", "heldout"):
        items = [x for x in report["samples"] if x["split"] == split]
        print(
            "EVAL",
            report["stage_size"],
            report["stage_step"],
            split,
            sum(x["passed"] for x in items),
            "/",
            len(items),
            flush=True,
        )
    for item in report["samples"]:
        if item["split"] == "train":
            print("ANSWER", item["row"], item["seed"], repr(item["text"]), item["passed"], flush=True)
            if report.get("suite") == "length":
                print("FREE_BLOCKS", item["row"], item["seed"], item["blocks"], flush=True)
    if report.get("suite") == "length":
        for item in report["recovery"]:

            def ratio(numerator, denominator):
                return round(numerator / denominator, 4) if denominator else None

            print(
                "BLOCK_RECOVERY",
                item["split"],
                item["row"],
                item["block"],
                item["noise"],
                {
                    "body": ratio(item["body_restored"], item["body_changed"]),
                    "eos_tail": ratio(item["tail_restored"], item["tail_changed"]),
                    "body_CE": ratio(item["body_ce_sum"], item["body_count"]),
                    "tail_CE": ratio(item["tail_ce_sum"], item["tail_count"]),
                },
                flush=True,
            )
    for item in report.get("teacher_blocks", []):
        print("CORRECT_PREFIX_BLOCK", item["row"], item["block"], item["seed"], item["result"], flush=True)


def main():
    args = arguments()
    out = Path(args.out)
    if args.summary_only:
        print((out / "summary.json").read_text(encoding="utf-8"))
        for path in sorted(out.glob("stage-*/eval-*.json")):
            report = json.loads(path.read_text())
            print_report(report)
        return
    fresh_destination(out)
    hardware = preflight(args.tiny, out)
    print("PREFLIGHT", hardware, flush=True)
    signature = source_hash()
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "resume", "summary_only")}
    settings.update(
        lr=1.125e-4,
        warmup=50,
        canvas=16 if args.tiny else 256,
        max_canvases=18,
        loss_mode="canvas-mean",
        optimizer="bf16-adamw",
        code_sha256=signature,
        gate="all active training cases pass every seed twice consecutively",
    )
    restored = None
    if args.resume:
        if not (args.resume / "COMPLETE").exists():
            raise RuntimeError("Incomplete checkpoint; refusing to resume")
        # Only load a checkpoint the user created locally; this includes Python RNG state.
        restored = torch.load(args.resume / "diagnostic_state.pt", map_location="cpu", weights_only=False)
        if restored["settings"] != settings:
            raise RuntimeError("Checkpoint settings/code differ; no resume started")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "settings.json", settings)
    random.seed(42)
    torch.manual_seed(42)
    canvas = settings["canvas"]
    if args.tiny:
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = Gemma4ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    prepare_conversion_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(0.95, 0.99), weight_decay=1e-4)
    train, heldout = task_rows(args.suite)
    if args.tiny:
        if args.suite == "length":
            train = [
                dict(train[0], prompt="short", response="abc"),
                dict(train[1], prompt="long", response="abcdefghijklmno0123456789"),
            ]
            heldout = [dict(heldout[0], prompt="unseen", response="xyz")]
        else:
            train = [dict(r, prompt=f"sum {i}", response=str(i + 5)) for i, r in enumerate(train)]
            heldout = [dict(r, prompt=f"unseen {i}", response=str(i + 13)) for i, r in enumerate(heldout)]
    metadata = length_metadata(tokenizer, train, canvas) if args.suite == "length" and not args.tiny else None
    write_json(out / "tasks.json", {"train": train, "heldout": heldout, "lengths": metadata})
    print("TASKS", metadata if metadata else args.suite, flush=True)
    global_step, start_stage, offset, streak = 0, 0, 0, 0
    completed = []
    if restored:
        load_model_weights(model, args.resume)
        _get_conditioner(model).load_state_dict(
            load_file(str(args.resume / "self_conditioner.safetensors")), strict=True
        )
        optimizer.load_state_dict(restored["optimizer"])
        global_step, start_stage, offset, streak = [
            restored[k] for k in ("global_step", "stage_index", "stage_step", "streak")
        ]
        completed = restored["completed"]
        torch.set_rng_state(restored["cpu_rng"])
        if restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        random.setstate(restored["python_rng"])
    device = next(model.parameters()).device
    encoded = [encode(tokenizer, r, device, canvas * 18) for r in train]
    summary = {"status": "running", "completed_stages": completed, "conversion_success": "not assessed"}
    write_json(out / "summary.json", summary)
    for stage_index in range(start_stage, len(args.stages)):
        n = args.stages[stage_index]
        stage_dir = out / f"stage-{n}"
        stage_dir.mkdir(exist_ok=True)
        stage_step = offset if stage_index == start_stage else 0
        current_streak = streak if stage_index == start_stage else 0
        if stage_step == 0:
            with preserve_rng():
                report = evaluate(model, tokenizer, train[:n], heldout, args, canvas)
            report.update(stage_size=n, stage_step=0, global_step=global_step, suite=args.suite)
            write_json(stage_dir / "eval-0000.json", report)
            print_report(report)
        log_name = f"train-resume-{global_step}.jsonl" if args.resume and stage_index == start_stage else "train.jsonl"
        with (stage_dir / log_name).open("a", encoding="utf-8") as log:
            while stage_step < args.steps_per_stage and current_streak < 2:
                started = time.perf_counter()
                stage_step += 1
                global_step += 1
                index = (stage_step - 1) % n
                torch.manual_seed(420000 + global_step)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr = settings["lr"] * min(global_step / 50, 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                prompt, target = encoded[index]
                loss, parts = diffusion_batch_step(
                    model,
                    [prompt],
                    [target],
                    canvas_size=canvas,
                    num_canvases=18,
                    pad_id=0,
                    eos_id=1,
                    diagnostics=True,
                    loss_mode="canvas-mean",
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss; next stage not started")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient; next stage not started")
                optimizer.step()
                record = {
                    "step": global_step,
                    "stage_size": n,
                    "stage_step": stage_step,
                    "row": index,
                    "loss": float(loss.detach()),
                    "grad": float(norm),
                    "lr": lr,
                    "seconds": time.perf_counter() - started,
                    "diagnostics": parts["rows"][0],
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if stage_step == 1 or stage_step % 10 == 0:
                    print("TRAIN", json.dumps(record), flush=True)
                if stage_step % args.eval_every == 0 or stage_step == args.steps_per_stage:
                    with preserve_rng():
                        report = evaluate(model, tokenizer, train[:n], heldout, args, canvas)
                    report.update(stage_size=n, stage_step=stage_step, global_step=global_step, suite=args.suite)
                    write_json(stage_dir / f"eval-{stage_step:04d}.json", report)
                    print_report(report)
                    current_streak = stable_pass(report, current_streak)
                    state = {
                        "global_step": global_step,
                        "stage_index": stage_index,
                        "stage_step": stage_step,
                        "streak": current_streak,
                        "completed": completed.copy(),
                        "settings": settings,
                    }
                    save_checkpoint(model, tokenizer, optimizer, out, state)
        if current_streak < 2:
            summary.update(status="stopped_at_failed_stage", failed_stage=n, global_step=global_step)
            write_json(out / "summary.json", summary)
            print("STOP_NO_ADVANCE", n, "Stable generation gate not met; no larger experiment started.", flush=True)
            return
        completed.append(n)
        summary.update(completed_stages=completed.copy(), global_step=global_step)
        write_json(out / "summary.json", summary)
        print("STAGE_PASSED", n, "Training-set fit only; heldout is reported separately.", flush=True)
    summary.update(status="passed_all_requested_stages", global_step=global_step)
    write_json(out / "summary.json", summary)
    print("SINGLE_SAMPLE_DIAGNOSTIC_DONE", out, flush=True)


if __name__ == "__main__":
    main()
