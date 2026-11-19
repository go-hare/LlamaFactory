#!/usr/bin/env python3
"""Stream-convert Gemma-4 diffusion JSONL splits to Qwen3 ChatML and optionally upload to HF Hub."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TURN_RE = re.compile(r"<\|turn>(system|user|model)\n(.*?)<turn\|>", re.DOTALL)


def gemma_to_qwen3_texts(raw_prompt: str, raw_response: str) -> tuple[str, str]:
    """Convert Gemma-4 wrapped prompt/response into Qwen3 ChatML + <think> tags."""
    if "<|im_start|>" in raw_prompt and raw_response.endswith("<|im_end|>"):
        return raw_prompt, raw_response
    cleaned_prompt = raw_prompt.removeprefix("<bos>")
    parts: list[str] = []
    for role, body in _TURN_RE.findall(cleaned_prompt):
        q_role = "assistant" if role == "model" else role
        if q_role == "system":
            body = body.replace("<|think|>\n", "").replace("<|think|>", "")
        parts.append(f"<|im_start|>{q_role}\n{body}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    q_prompt = "".join(parts)

    resp = raw_response
    if resp.endswith("<turn|>"):
        resp = resp[: -len("<turn|>")]
    if resp.startswith("<|channel>thought\n"):
        resp = resp[len("<|channel>thought\n") :]
        if "<channel|>" in resp:
            thought, final = resp.split("<channel|>", 1)
            resp = f"<think>\n{thought}\n</think>\n{final.lstrip()}"
        else:
            resp = f"<think>\n{resp}"
    q_response = f"{resp}<|im_end|>"
    return q_prompt, q_response


def convert_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            q_prompt, q_response = gemma_to_qwen3_texts(row["prompt"], row["response"])
            out_row = {"prompt": q_prompt, "response": q_response, "bucket": int(row.get("bucket", 0))}
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            count += 1
            if count % 50000 == 0:
                print(f"Converted {src.name}: {count} rows", flush=True)
    print(f"Done {src.name}: {count} rows -> {dst}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    for split in ("valid.jsonl", "train.jsonl"):
        src_file = args.src / split
        if src_file.is_file():
            convert_file(src_file, args.dst / split)

    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        print(f"Uploading {args.dst} to hf://datasets/{args.repo_id} (private={not args.public}) ...", flush=True)
        api.upload_folder(
            folder_path=str(args.dst),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Upload Qwen3 ChatML converted diffusion_600m dataset",
        )
        print(f"UPLOAD_COMPLETE https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
