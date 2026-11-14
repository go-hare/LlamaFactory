#!/usr/bin/env bash
# Sweep 3 learning rates (3e-5, 6e-5, 1.125e-4) for 400 steps each on the 5,000-row
# probe split, then evaluate validation loss and generate 5 sample responses using
# Official Algorithm 1 (entropy-bounded sampler with self-conditioning & early stop).
#
# Usage on DGX Spark (128GB) or AMD MI300 (192GB):
#   DATA_DIR=/workspace/diffusion_600m OUT_ROOT=/workspace/saves/lr_sweep bash scripts/sweep_gemma4_diffusion_lr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

export DISABLE_VERSION_CHECK=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-google/gemma-4-E2B-it}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/diffusion_600m}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/saves/gemma4_e2b_lr_sweep}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_STEPS="${MAX_STEPS:-400}"
EVAL_STEPS="${EVAL_STEPS:-200}"

mkdir -p "${OUT_ROOT}"

for LR in "3.0e-5" "6.0e-5" "1.125e-4"; do
  TAG="lr_${LR}"
  RUN_DIR="${OUT_ROOT}/${TAG}"
  echo "============================================================"
  echo "[Sweep] Starting ${TAG} -> ${RUN_DIR}"
  echo "============================================================"

  python3 -m llamafactory.cli train \
    --stage diffusion \
    --do_train true \
    --do_eval true \
    --model_name_or_path "${MODEL_PATH}" \
    --trust_remote_code true \
    --finetuning_type full \
    --dataset_dir "${DATA_DIR}" \
    --dataset probe_train \
    --eval_dataset probe_valid \
    --cutoff_len 1024 \
    --output_dir "${RUN_DIR}" \
    --overwrite_output_dir true \
    --save_only_model true \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --weight_decay 1.0e-4 \
    --adam_beta1 0.95 \
    --adam_beta2 0.99 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --max_steps "${MAX_STEPS}" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${MAX_STEPS}" \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none \
    --rl_steps 0 \
    2>&1 | tee "${RUN_DIR}.log"

  echo "[Sweep] Sampling 5 test prompts on ${TAG}..."
  python3 - "${RUN_DIR}" <<'PY'
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llamafactory.train.diffusion.loss import SelfConditioner
from llamafactory.train.diffusion.rl import sample_response
from llamafactory.train.diffusion.workflow import _load_conditioner_weights

ckpt = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()

text_cfg = getattr(model.config, "text_config", model.config)
hidden = int(getattr(text_cfg, "hidden_size", 0) or model.config.hidden_size)
inter = int(getattr(text_cfg, "intermediate_size", 0) or hidden * 4)
cond = SelfConditioner(hidden, inter).to(device=device, dtype=torch.bfloat16)
_load_conditioner_weights(cond, ckpt)
model.self_conditioner = cond

prompts = [
    "<bos><|turn>system\n你是一个温和、真诚的朋友。<turn|>\n<|turn>user\n最近加班太多，感觉整个人都被掏空了。<turn|>\n<|turn>model\n",
    "<bos><|turn>user\n解方程：3x + 7 = 22，写出步骤。<turn|>\n<|turn>model\n",
    "<bos><|turn>user\n用 Python 写一个二分查找函数。<turn|>\n<|turn>model\n",
    "<bos><|turn>user\n把这句话润色得更自然：由于天气不好的原因，我们取消了原本计划好的出行。<turn|>\n<|turn>model\n",
    "<bos><|turn>user\nBriefly explain why the sky appears blue.<turn|>\n<|turn>model\n",
]

for i, p in enumerate(prompts, 1):
    ids = torch.tensor(tokenizer.encode(p, add_special_tokens=False), dtype=torch.long, device=device)
    with torch.no_grad():
        out = sample_response(
            model,
            ids,
            max_canvases=2,
            steps=48,
            canvas_size=256,
            pad_id=tokenizer.pad_token_id or 0,
            stop_tokens=(tokenizer.eos_token_id or 1, 106),
        )
    print(f"\n--- Sample {i} ---")
    print(tokenizer.decode(out.tolist(), skip_special_tokens=False))
PY
done
