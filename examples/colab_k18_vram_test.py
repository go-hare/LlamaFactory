"""Colab Pro+ VRAM check for section-4 conversion: C=256, K=18, prompt 4096.

Runtime: A100 80GB (High-RAM). T4 / A100 40GB will fail. Stop after 5 steps.

  1. Runtime → Change runtime type → A100 High-RAM / 80GB
  2. Upload llamafactory_k18.tgz (src + probe jsonl, not train.jsonl)
  3. Paste HF token when asked
  4. Run this file as a notebook cell block, or: !python colab_k18_vram_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

STEPS = 5
TARBALL = Path("/content/llamafactory_k18.tgz")
ROOT = Path("/content/LlamaFactory")
DATA = ROOT / "data" / "diffusion_600m"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.check_call(cmd, shell=True)


def require_a100_80() -> None:
    out = subprocess.check_output(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        shell=True,
        text=True,
    ).strip()
    print("GPU:", out, flush=True)
    if "81920" not in out.replace(" ", "") and "80" not in out:
        raise SystemExit("Need A100 80GB. Current runtime is not 80GB. Stop.")


def main() -> None:
    require_a100_80()
    if not TARBALL.is_file():
        raise SystemExit(f"Upload {TARBALL} first (code + probe_train/probe_valid only).")
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL) as tar:
        tar.extractall(ROOT)
    os.environ["DISABLE_VERSION_CHECK"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise SystemExit("Set HF_TOKEN in the notebook before running.")
    sh('pip install -q "transformers>=5.17.0" peft accelerate datasets safetensors sentencepiece einops fire omegaconf "tyro<0.9.0" protobuf pyyaml pydantic packaging tiktoken hf_transfer')
    sys.path.insert(0, str(ROOT / "src"))
    os.chdir(ROOT)
    cmd = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        "--stage",
        "diffusion",
        "--do_train",
        "true",
        "--model_name_or_path",
        os.environ.get("MODEL_PATH", "google/gemma-4-E2B-it"),
        "--trust_remote_code",
        "true",
        "--finetuning_type",
        "full",
        "--dataset_dir",
        str(DATA),
        "--dataset",
        "probe_train",
        "--cutoff_len",
        "4096",
        "--output_dir",
        "/content/saves/k18_test",
        "--overwrite_output_dir",
        "true",
        "--save_only_model",
        "true",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "1.125e-4",
        "--max_steps",
        str(STEPS),
        "--logging_steps",
        "1",
        "--eval_strategy",
        "no",
        "--bf16",
        "true",
        "--pure_bf16",
        "true",
        "--gradient_checkpointing",
        "true",
        "--report_to",
        "none",
        "--rl_steps",
        "0",
    ]
    print("TRAIN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv")
    print("K18_OK", flush=True)


if __name__ == "__main__":
    main()
