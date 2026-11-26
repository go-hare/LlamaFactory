#!/usr/bin/env python3
"""
一键将本地构建好的路线一黄金数据集 (`data/gemma4_duplex_route1`) 上传至你的 Hugging Face Dataset 仓库
默认目标仓库: kongDeQiang/gemma4-duplex-route1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="上传路线一全双工数据集到 Hugging Face Hub")
    p.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "gemma4_duplex_route1"))
    p.add_argument("--hf-repo", type=str, default="kongDeQiang/gemma4-duplex-route1")
    p.add_argument("--private", action="store_true", default=False, help="是否设为私有数据集仓库")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    train_p = data_dir / "train.json"
    valid_p = data_dir / "valid.json"
    if not train_p.exists() or not valid_p.exists():
        raise FileNotFoundError(f"未找到 train.json / valid.json，请先运行 scripts/build_golden_duplex_dataset.py: {data_dir}")

    train_n = len(json.loads(train_p.read_text(encoding="utf-8")))
    valid_n = len(json.loads(valid_p.read_text(encoding="utf-8")))
    wav_cnt = len(list((data_dir / "audios").glob("*.wav"))) if (data_dir / "audios").exists() else 0

    readme_md = f"""---
license: apache-2.0
task_categories:
  - audio-text-to-text
  - text-generation
language:
  - zh
tags:
  - full-duplex
  - gemma-4
  - tool-calling
  - speech
---

# Gemma-4 Route-1 Full-Duplex Native Audio + Async Tool-Calling Dataset

- **Train samples**: {train_n}
- **Valid samples**: {valid_n}
- **16kHz acoustic clips (`audios/*.wav`)**: {wav_cnt} (AudioSet environmental noise, MIT RIR far-field reverberant bystander speech, and multi-speaker near-field Mandarin utterances)
- **Control token slots**: `<unused0>` ~ `<unused7>` (Gemma-4 native IDs `6..13`, zero vocab expansion)
"""
    (data_dir / "README.md").write_text(readme_md, encoding="utf-8")

    api = HfApi()
    user = api.whoami().get("name")
    print(f"[HF] 当前登录账号: {user} | 目标数据集仓库: {args.hf_repo} (private={args.private})")
    api.create_repo(repo_id=args.hf_repo, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"[HF] 正在上传 {data_dir} (train={train_n}, valid={valid_n}, wavs={wav_cnt})...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.hf_repo,
        repo_type="dataset",
        ignore_patterns=[".gitignore", ".DS_Store", "tmp_*", "*.aiff"],
        commit_message=f"Upload Gemma-4 Route-1 full-duplex dataset (train={train_n}, valid={valid_n}, wavs={wav_cnt})",
    )
    print(f"✅ 数据集已成功上传至: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
