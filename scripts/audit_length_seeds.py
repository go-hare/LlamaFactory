#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Fixed checkpoint, seen prompts, previously unused noise seeds. No training."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_single_sample_controls import clean_answer, compare_blocks, fresh_destination, source_hash, write_json
from validate_diffusion_conversion import ByteTokenizer, encode

from llamafactory.train.diffusion.loss import _get_conditioner
from llamafactory.train.diffusion.rl import sample_response


def validate_plan(checkpoint, seeds, tiny):
    if not (checkpoint / "COMPLETE").is_file():
        raise RuntimeError("Checkpoint is incomplete; no model loaded")
    root = checkpoint.parent
    settings = json.loads((root / "settings.json").read_text())
    if settings.get("suite") != "length":
        raise RuntimeError("This audit requires the saved length-suite experiment")
    if settings.get("code_sha256") != source_hash():
        raise RuntimeError("Training source hash differs. Use the exact length-suite runtime source.")
    if bool(settings.get("tiny")) != tiny:
        raise RuntimeError("Tiny/real checkpoint mode mismatch")
    old_seeds = set(settings["seeds"])
    if set(seeds) & old_seeds:
        raise ValueError(f"Audit seeds overlap old evaluation seeds: {sorted(set(seeds) & old_seeds)}")
    if len(seeds) != len(set(seeds)) or not seeds or any(s < 0 for s in seeds):
        raise ValueError("Use unique nonnegative seeds")
    tasks = json.loads((root / "tasks.json").read_text())["train"]
    if len(tasks) != 2:
        raise RuntimeError("Expected exactly the two saved training prompts")
    for name in ("config.json", "self_conditioner.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not ((checkpoint / "model.safetensors").exists() or (checkpoint / "model.safetensors.index.json").exists()):
        raise FileNotFoundError("No model weights in checkpoint")
    return settings, tasks


def preflight(tiny):
    if tiny:
        return {"mode": "CPU tiny checkpoint; execution checks only"}
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU; no model loaded")
    free, total = torch.cuda.mem_get_info()
    info = {"gpu": torch.cuda.get_device_name(), "free_gib": free / 1024**3, "total_gib": total / 1024**3}
    if total < 75 * 1024**3 or free < 20 * 1024**3:
        raise RuntimeError(f"Use the existing 80GB-class GPU with >=20 GiB free: {info}")
    return info


def score_result(row, seed, text, output, target, canvas, seconds):
    blocks = compare_blocks(output, target, canvas)
    for block in blocks:
        begin = block["block"] * canvas
        expected = target[begin : begin + block["expected_tokens"]]
        actual = output[begin : begin + block["expected_tokens"]]
        wrong = (actual != expected[: len(actual)]).nonzero().flatten().tolist()
        wrong += list(range(len(actual), len(expected)))
        block["wrong_count"] = len(wrong)
        block["first_wrong_offsets"] = wrong[:16]
    return {
        "row": row,
        "seed": seed,
        "text": text,
        "passed": clean_answer(text) == row["response"],
        "target_tokens": len(target),
        "output_tokens": len(output),
        "has_stop": bool(((output == 1) | (output == 106)).any()),
        "blocks": blocks,
        "seconds": seconds,
    }


def summarize(path):
    records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    summary = {
        "scope": "seen training prompts, new evaluation noise seeds; not generalization or a new training-gate pass",
        "attempts": len(records),
        "tasks": [],
    }
    for index in sorted({r["row_index"] for r in records}):
        group = [r for r in records if r["row_index"] == index]
        item = {
            "row_index": index,
            "kind": group[0]["row"]["kind"],
            "passed": sum(r["passed"] for r in group),
            "attempts": len(group),
            "blocks": [],
        }
        for block in sorted({b["block"] for r in group for b in r["blocks"]}):
            bs = [b for r in group for b in r["blocks"] if b["block"] == block]
            item["blocks"].append(
                {
                    "block": block,
                    "exact": sum(b["exact"] for b in bs),
                    "attempts": len(bs),
                    "wrong_tokens": sum(b["wrong_count"] for b in bs),
                }
            )
        summary["tasks"].append(item)
        print("TASK_RESULT", json.dumps(item, ensure_ascii=False), flush=True)
    for r in records:
        if not r["passed"]:
            print("FAILED_SEED", r["row_index"], r["seed"], r["blocks"], repr(r["text"][:160]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/content/drive/MyDrive/saves/e2b_length_controls_v1/checkpoint-800")
    )
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/e2b_length_seed_audit_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 120)))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        summarize(args.out / "samples.jsonl")
        return
    fresh_destination(args.out)
    settings, tasks = validate_plan(args.checkpoint, args.seeds, args.tiny)
    hardware = preflight(args.tiny)
    print("READ_ONLY_AUDIT", hardware, "seeds", args.seeds, flush=True)
    device = "cpu" if args.tiny else "cuda"
    tokenizer = ByteTokenizer() if args.tiny else AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model, info = Gemma4ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        local_files_only=True,
        output_loading_info=True,
    )
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("self_conditioner.")]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Checkpoint load mismatch: {info}")
    conditioner = _get_conditioner(model)
    conditioner.load_state_dict(load_file(str(args.checkpoint / "self_conditioner.safetensors")), strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    canvas, steps = settings["canvas"], settings["sample_steps"]
    if not args.tiny and (canvas != 256 or steps != 48):
        raise ValueError("Audit expects the original C=256, maximum 48 denoising-step settings")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": (args.checkpoint / "COMPLETE").read_text(),
        "seeds": args.seeds,
        "old_seeds": settings["seeds"],
        "canvas": canvas,
        "steps": steps,
        "max_canvases": 3,
        "scope": "fixed seen prompts; new noise seeds only; no optimizer or training",
        "training_code_sha256": settings["code_sha256"],
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(args.out / "settings.json", manifest)
    with (args.out / "samples.jsonl").open("w", encoding="utf-8") as log, torch.inference_mode():
        for index, row in enumerate(tasks):
            prompt, target = encode(tokenizer, row, device, canvas * 18)
            for seed in args.seeds:
                torch.manual_seed(seed)
                if not args.tiny:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = sample_response(
                    model,
                    prompt,
                    steps=steps,
                    canvas_size=canvas,
                    max_canvases=3,
                    vocab_size=model.config.text_config.vocab_size,
                )
                if not args.tiny:
                    torch.cuda.synchronize()
                text = tokenizer.decode(output.tolist(), skip_special_tokens=False)
                record = score_result(row, seed, text, output, target, canvas, time.perf_counter() - start)
                record["row_index"] = index
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print("SEED_RESULT", index, seed, record["passed"], record["blocks"], flush=True)
    summary = summarize(args.out / "samples.jsonl")
    write_json(args.out / "summary.json", summary)
    print("SEED_AUDIT_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
