#!/usr/bin/env python3
# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0

"""Select complete real answers into reproducible, normalized-deduplicated splits."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


BUCKETS = [(32, 256), (257, 512), (513, 1024), (1025, 4608)]
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
HASH_KEYS = ("prompt_hash", "question_hash", "response_hash")


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def digest(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def tokenizer_fingerprint(tokenizer):
    """Hash canonical backend JSON (or vocabulary for tokenizers without a backend)."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        payload = json.loads(backend.to_str())
        method = "backend_tokenizer.to_str"
    else:
        payload = tokenizer.get_vocab()
        method = "get_vocab"
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "name_or_path": str(tokenizer.name_or_path),
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "fingerprint_method": method,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def identities(row):
    users = re.findall(r"<\|turn>user\n([\s\S]*?)<turn\|>", row["prompt"])
    if not users:
        raise ValueError("No complete user turn")
    return {
        "prompt_hash": digest(row["prompt"]),
        "question_hash": digest(users[-1]),
        "response_hash": digest(row["response"]),
    }


def boundary_error(prompt, response):
    prefix = prompt.removeprefix("<bos>")
    if not prefix.endswith("<|turn>model\n"):
        return "missing_turn_boundary"
    history = prefix[: -len("<|turn>model\n")]
    turns = list(re.finditer(r"<\|turn>(system|user|model)\n((?:(?!<\|turn>|<turn\|>)[\s\S])*)<turn\|>\n?", history))
    if not turns or "".join(turn.group() for turn in turns) != history:
        return "malformed_prompt_boundary"
    if not any(turn.group(1) == "user" for turn in turns):
        return "no_user"
    if not response.endswith("<turn|>") or "<turn|>" in response[: -len("<turn|>")] or "<|turn>" in response:
        return "missing_turn_boundary"
    for text in (prompt, response):
        in_thought = False
        for marker in re.findall(r"<\|channel>thought|<channel\|>", text):
            if marker == "<|channel>thought":
                if in_thought:
                    return "unclosed_thought"
                in_thought = True
            else:
                if not in_thought:
                    return "malformed_thought"
                in_thought = False
        if in_thought:
            return "unclosed_thought"
    return None


def bucket_for(length):
    return next((i for i, (low, high) in enumerate(BUCKETS) if low <= length <= high), None)


def positive(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def choose(path, tokenizer, per_bucket, used, prompt_cap=4096):
    positive(per_bucket, "per_bucket")
    positive(prompt_cap, "prompt_cap")
    if prompt_cap > 4096:
        raise ValueError("prompt_cap must be <= 4096")
    selected, counts, skipped = [], Counter(), Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            p, r = (row.get("prompt"), row.get("response")) if isinstance(row, dict) else (None, None)
            if not isinstance(p, str) or not isinstance(r, str) or not r.strip():
                skipped["empty_or_invalid"] += 1
                continue
            error = boundary_error(p, r)
            if error:
                skipped[error] += 1
                continue
            hashes = identities(row)
            if any(h in used[k] for k, h in hashes.items()):
                skipped["normalized_duplicate"] += 1
                continue
            response_ids = tokenizer.encode(r, add_special_tokens=False, truncation=False)
            bucket = bucket_for(len(response_ids))
            if bucket is None:
                skipped["response_out_of_range"] += 1
                continue
            if counts[bucket] >= per_bucket:
                skipped["bucket_full"] += 1
                continue
            prompt_ids = tokenizer.encode(p, add_special_tokens=False, truncation=False)
            if not prompt_ids or len(prompt_ids) > prompt_cap:
                skipped["prompt_over_cap"] += 1
                continue
            record = {
                "prompt": p,
                "response": r,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "bucket": bucket,
                "source_split": Path(path).stem,
                "source_line": line_number,
                "thought": "<|channel>thought" in r,
                "code_fence": "```" in r,
                **hashes,
            }
            selected.append(record)
            counts[bucket] += 1
            for k, h in hashes.items():
                used[k].add(h)
            if all(counts[i] == per_bucket for i in range(len(BUCKETS))):
                break
    if len(selected) != per_bucket * len(BUCKETS):
        raise ValueError(f"Insufficient complete answers in {path}: {dict(counts)}")
    return selected, dict(skipped)


def write_dataset(train_path, valid_path, tokenizer, out, train_per=128, valid_per=16, prompt_cap=4096, C=256, K=18):
    for name, value in (
        ("train_per", train_per),
        ("valid_per", valid_per),
        ("prompt_cap", prompt_cap),
        ("C", C),
        ("K", K),
    ):
        positive(value, name)
    if prompt_cap > 4096:
        raise ValueError("prompt_cap must be <= 4096")
    if C * K < BUCKETS[-1][1]:
        raise ValueError("C * K must cover the complete 4608-token target budget")
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {out}")
    used = {k: set() for k in HASH_KEYS}
    valid, vs = choose(valid_path, tokenizer, valid_per, used, prompt_cap)
    train, ts = choose(train_path, tokenizer, train_per, used, prompt_cap)
    manifest = {
        "scope": "real-data pilot; not a full corpus or exact reproduction of Google data",
        "sampling": "valid first; deterministic source order; equal answer-length buckets; no text truncation",
        "dedup": "NFKC, remove whitespace, casefold: full prompt, last user turn, full response; no semantic near-dedup",
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "C": C,
        "K": K,
        "prompt_cap": prompt_cap,
        "buckets": BUCKETS,
        "splits": {},
    }
    contents = {}
    for name, rows, skip in [("train", train, ts), ("valid", valid, vs)]:
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        contents[name] = content
        lens = sorted(len(r["response_ids"]) for r in rows)
        prompts = sorted(len(r["prompt_ids"]) for r in rows)
        manifest["splits"][name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "bucket_counts": dict(Counter(r["bucket"] for r in rows)),
            "thought_rows": sum(r["thought"] for r in rows),
            "code_fence_rows": sum(r["code_fence"] for r in rows),
            "response_min": min(lens),
            "response_median": lens[len(lens) // 2],
            "response_max": max(lens),
            "response_tokens": sum(lens),
            "response_blocks_max": (max(lens) + C - 1) // C,
            "prompt_min": min(prompts),
            "prompt_median": prompts[len(prompts) // 2],
            "prompt_max": max(prompts),
            "prompt_tokens": sum(prompts),
            "joint_max": max(len(r["prompt_ids"]) + len(r["response_ids"]) for r in rows),
            "skipped": skip,
        }
    # Selection and metadata must succeed before any output is created. mkdir is exclusive.
    out.mkdir(parents=True)
    for name, content in contents.items():
        (out / f"{name}.jsonl").write_text(content, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-per-bucket", type=int, default=128)
    parser.add_argument("--valid-per-bucket", type=int, default=16)
    parser.add_argument("--prompt-cap", type=int, default=4096)
    parser.add_argument("--C", type=int, default=256)
    parser.add_argument("--K", type=int, default=18)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = write_dataset(
        args.src / "train.jsonl",
        args.src / "valid.jsonl",
        tokenizer,
        args.out,
        args.train_per_bucket,
        args.valid_per_bucket,
        args.prompt_cap,
        args.C,
        args.K,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
