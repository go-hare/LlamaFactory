#!/usr/bin/env python3
"""Convert Fast-dLLM Gemma 4 conversation shards into LlamaFactory Diffusion JSONL.

Reads ``Fast-dLLM/v2/data/gemma4/train_conversation/train_*.json`` (559k multi-turn
bilingual dialogues across zh_human_rp, zh_r1_cot, math_code_cot, agent_tool,
and multilingual) and writes ``{"prompt", "response"}`` JSONL files formatted with
Gemma 4's native ``<|turn>`` / ``<turn|>`` tags:

- ``train.jsonl``: full training split
- ``valid.jsonl``: held-out validation split (default 2,000 rows)
- ``probe_train.jsonl``: deterministic 5,000-row subset for learning-rate sweeps
- ``probe_valid.jsonl``: first 200 rows of valid, for fast eval during sweeps
- ``dataset_info.json``: LlamaFactory dataset registry

Usage:
    python scripts/prepare_diffusion_600m.py \
        --src /Users/apple/work-py/hare-code/Fast-dLLM/v2/data/gemma4/train_conversation \
        --dst /Users/apple/work-py/hare-code/LlamaFactory/data/diffusion_600m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


def format_instance(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    """Format a multi-turn ``messages`` list into ``(prompt, response)`` for Gemma 4.

    All turns up to the final user turn (plus ``<|turn>model\\n``) form ``prompt``;
    the final assistant turn (closed by ``<turn|>``) forms ``response``.
    """
    if not messages or len(messages) < 2:
        return None
    last_ass_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant" and (messages[idx].get("content") or "").strip():
            last_ass_idx = idx
            break
    if last_ass_idx <= 0:
        return None

    prefix = messages[:last_ass_idx]
    target = (messages[last_ass_idx].get("content") or "").strip()
    if not target:
        return None

    parts: list[str] = ["<bos>"]
    for i, msg in enumerate(prefix):
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if i == 0 and role == "system":
            parts.append(f"<|turn>system\n{content}<turn|>\n")
        elif role == "user":
            parts.append(f"<|turn>user\n{content}<turn|>\n")
        elif role == "assistant":
            parts.append(f"<|turn>model\n{content}<turn|>\n")
    parts.append("<|turn>model\n")
    prompt = "".join(parts)
    if "<|turn>user\n" not in prompt:
        return None
    response = target if target.endswith("<turn|>") else f"{target}<turn|>"
    return prompt, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/Users/apple/work-py/hare-code/Fast-dLLM/v2/data/gemma4/train_conversation",
        help="Directory containing train_*.json conversation shards",
    )
    parser.add_argument(
        "--dst",
        default="/Users/apple/work-py/hare-code/LlamaFactory/data/diffusion_600m",
        help="Output directory for train.jsonl, valid.jsonl, probe_train.jsonl",
    )
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--probe-size", type=int, default=5000)
    parser.add_argument("--probe-valid-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.src, "train_*.json")))
    if not shard_paths:
        raise FileNotFoundError(f"No train_*.json shards found in {args.src}")

    out_dir = Path(args.dst)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    all_rows: list[tuple[str, str]] = []
    for sp in shard_paths:
        with open(sp, encoding="utf-8") as f:
            payload = json.load(f)
        instances = payload["instances"] if isinstance(payload, dict) else payload
        for item in instances:
            pair = format_instance(item.get("messages") or [])
            if pair is not None:
                all_rows.append(pair)
    print(f"Loaded {len(all_rows)} valid dialogues from {len(shard_paths)} shards.", flush=True)

    rng.shuffle(all_rows)
    valid_rows = all_rows[: args.valid_size]
    train_rows = all_rows[args.valid_size :]
    probe_rows = train_rows[: args.probe_size]
    probe_valid_rows = valid_rows[: args.probe_valid_size]

    def write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8") as out:
            for p, r in rows:
                out.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "probe_train.jsonl", probe_rows)
    write_jsonl(out_dir / "probe_valid.jsonl", probe_valid_rows)

    dataset_info = {
        "train": {
            "file_name": "train.jsonl",
            "columns": {"prompt": "prompt", "response": "response"},
        },
        "valid": {
            "file_name": "valid.jsonl",
            "columns": {"prompt": "prompt", "response": "response"},
        },
        "probe_train": {
            "file_name": "probe_train.jsonl",
            "columns": {"prompt": "prompt", "response": "response"},
        },
        "probe_valid": {
            "file_name": "probe_valid.jsonl",
            "columns": {"prompt": "prompt", "response": "response"},
        },
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote train={len(train_rows)}, valid={len(valid_rows)}, probe_train={len(probe_rows)}, probe_valid={len(probe_valid_rows)} to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
