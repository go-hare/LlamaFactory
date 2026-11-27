#!/usr/bin/env python3
"""
一键打包并上传路线二 (II-A) Stage 1 + Stage 2 纯净无毒数据集到 Hugging Face Hub (`kongDeQiang/gemma4-route2-omni-dataset`)。
采用单文件 `audios.tar` 无损 FLAC 归档避免数千小文件触发 HF HTTP 429 限流，Colab 下载解压仅需 ~5 秒。
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-repo", type=str, default="kongDeQiang/gemma4-route2-omni-dataset")
    p.add_argument("--private", action="store_true", default=False)
    args = p.parse_args()

    s1_dir = REPO_ROOT / "data" / "gemma4_route2_stage1"
    s2_dir = REPO_ROOT / "data" / "gemma4_route2_stage2"
    stage_upload_dir = REPO_ROOT / "data" / "_route2_hf_staging"
    stage_upload_dir.mkdir(parents=True, exist_ok=True)

    s1_train_n = len(json.loads((s1_dir / "train.json").read_text(encoding="utf-8")))
    s1_valid_n = len(json.loads((s1_dir / "valid.json").read_text(encoding="utf-8")))
    s2_train_n = len((s2_dir / "train.jsonl").read_text(encoding="utf-8").splitlines())
    s2_valid_n = len((s2_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines())
    flacs = sorted((s1_dir / "audios").glob("*.flac"))

    tar_path = stage_upload_dir / "route2_bundle.tar"
    print(f"📦 Packing {len(flacs)} lossless 0.48s FLAC chunks + Stage1/Stage2 JSONs into {tar_path}...")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(s1_dir / "train.json", arcname="gemma4_route2_stage1/train.json")
        tf.add(s1_dir / "valid.json", arcname="gemma4_route2_stage1/valid.json")
        tf.add(s1_dir / "audios", arcname="gemma4_route2_stage1/audios")
        tf.add(s2_dir / "train.jsonl", arcname="gemma4_route2_stage2/train.jsonl")
        tf.add(s2_dir / "valid.jsonl", arcname="gemma4_route2_stage2/valid.jsonl")

    readme = f"""---
license: apache-2.0
language:
  - zh
tags:
  - full-duplex
  - speech-to-speech
  - gemma-4
  - cosyvoice2
---

# Gemma-4-E2B Route 2 (II-A) Native Speech-to-Speech Full-Duplex Dataset

- **Stage 1 (Thinker Full-Duplex + 50% Tool-Irrelevance + Real Human Dual-Track Rhythm)**: {s1_train_n} train / {s1_valid_n} valid ({len(flacs)} physical `0.48s` @ `16kHz` FLAC frames)
- **Stage 2 (Single-Speaker 25Hz CosyVoice2 Talker Alignment)**: {s2_train_n} train / {s2_valid_n} valid (`zh-CN-XiaoxiaoNeural` single voice)
"""
    (stage_upload_dir / "README.md").write_text(readme, encoding="utf-8")

    api = HfApi()
    user = api.whoami().get("name")
    print(f"[HF] User: {user} -> Repo: {args.hf_repo}")
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(stage_upload_dir),
        repo_id=args.hf_repo,
        repo_type="dataset",
        commit_message=f"Upload Route 2 Stage1 ({s1_train_n}+{s1_valid_n}) & Stage2 ({s2_train_n}+{s2_valid_n}) pure dataset",
    )
    print(f"✅ Uploaded to https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
