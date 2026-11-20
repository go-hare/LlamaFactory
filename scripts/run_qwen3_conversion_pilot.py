#!/usr/bin/env python3
"""Section-4 AR-to-diffusion conversion pilot for Qwen3 (e.g., Qwen/Qwen3-8B)."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_single_sample_controls import fresh_destination, load_model_weights, preflight, save_checkpoint, write_json
from validate_diffusion_conversion import ByteTokenizer, preserve_rng

from llamafactory.train.diffusion.loss import (
    _get_conditioner,
    denoise_logits,
    diffusion_all_blocks_step,
    diffusion_batch_step,
    prepare_conversion_training,
)
from llamafactory.train.diffusion.rl import sample_response


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


def resolve_special_ids(tokenizer) -> tuple[int, int, tuple[int, ...]]:
    """Pick distinct pad_id and eos_id so EOS-tail canvas fill never collides with PAD."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    im_end = vocab.get("<|im_end|>")
    endoftext = vocab.get("<|endoftext|>")
    pad_id = endoftext if isinstance(endoftext, int) else (tokenizer.pad_token_id or 0)
    eos_id = im_end if isinstance(im_end, int) else (tokenizer.eos_token_id or 1)
    if eos_id == pad_id:
        eos_id = 1 if pad_id != 1 else 2
    stop_tokens = tuple(sorted({int(x) for x in (eos_id, tokenizer.eos_token_id) if x is not None and int(x) != pad_id}))
    if not stop_tokens:
        stop_tokens = (eos_id,)
    return int(pad_id), int(eos_id), stop_tokens


def resolve_mask_id(tokenizer, vocab_size: int, *, pad_id: int, eos_id: int) -> int:
    """Resolve a dedicated absorbing [MASK] token id without colliding with pad/eos."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for tok in ("<|mask|>", "<mask>", "[MASK]", "<|fim_middle|>"):
        tid = vocab.get(tok)
        if isinstance(tid, int) and tid not in (pad_id, eos_id):
            return tid
    candidate = vocab_size - 1
    while candidate in (pad_id, eos_id) and candidate > 0:
        candidate -= 1
    return candidate


class LazyQwen3Split:
    """Byte-offset lazy JSONL split so multi-GB datasets load with minimal CPU RAM."""

    def __init__(self, path: Path, tokenizer, *, prompt_cap: int = 4096, response_cap: int = 18 * 256):
        self.path = path
        self.tokenizer = tokenizer
        self.prompt_cap = prompt_cap
        self.response_cap = response_cap
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw = json.loads(handle.readline().decode("utf-8"))
        q_prompt, q_response = gemma_to_qwen3_texts(raw["prompt"], raw["response"])
        p_ids = self.tokenizer.encode(q_prompt, add_special_tokens=False)[: self.prompt_cap]
        r_ids = self.tokenizer.encode(q_response, add_special_tokens=False)[: self.response_cap]
        if not p_ids or not r_ids:
            p_ids = p_ids or [self.tokenizer.bos_token_id or 151643]
            r_ids = r_ids or [self.tokenizer.eos_token_id or 151645]
        return {
            "prompt": q_prompt,
            "response": q_response,
            "prompt_ids": p_ids,
            "response_ids": r_ids,
            "bucket": int(raw.get("bucket", 0)),
        }


def load_qwen3_splits(directory: Path, tokenizer, *, prompt_cap: int = 4096, response_cap: int = 18 * 256):
    return {
        split: LazyQwen3Split(directory / f"{split}.jsonl", tokenizer, prompt_cap=prompt_cap, response_cap=response_cap)
        for split in ("valid", "train")
    }


def fixed_plan(splits, canvas: int):
    plan = {"recovery": [], "generation": []}
    for split, rows in splits.items():
        probe_limit = min(len(rows), 256)
        probed = [(i, rows[i]) for i in range(probe_limit)]
        probed_map = dict(probed)
        selected = []
        for bucket in range(4):
            indexes = [i for i, r in probed if r["bucket"] == bucket]
            if not indexes:
                continue
            if split == "valid":
                selected.extend(indexes[:64])
                plan["generation"].append({"split": split, "row": indexes[0], "seed": 100})
            else:
                subset = indexes[:4]
                longest_index = max(indexes, key=lambda i: len(probed_map[i]["response_ids"]))
                if longest_index not in subset:
                    subset[-1] = longest_index
                selected.extend(subset)
        if split == "valid" and len(plan["generation"]) < 4:
            for i in selected[:4]:
                if not any(g["split"] == split and g["row"] == i for g in plan["generation"]):
                    plan["generation"].append({"split": split, "row": i, "seed": 100})
                if len(plan["generation"]) >= 4:
                    break
        selected = selected[: (64 if split == "valid" else 16)]
        longest = max(selected, key=lambda i: len(probed_map[i]["response_ids"]))
        for offset, index in enumerate(selected):
            blocks = (len(probed_map[index]["response_ids"]) + canvas - 1) // canvas
            block = (0, (blocks - 1) // 2, blocks - 1)[offset % 3]
            if index == longest:
                block = blocks - 1
            plan["recovery"].append({"split": split, "row": index, "block": block})
    return plan


def tensor_row(row, device):
    return (
        torch.tensor(row["prompt_ids"], dtype=torch.long, device=device),
        torch.tensor(row["response_ids"], dtype=torch.long, device=device),
    )


def sample_pilot_response(model, prompt, settings, *, pad_id: int, stop_tokens: tuple[int, ...]):
    started = time.perf_counter()
    chunks, history = [], prompt
    reason = "canvas_budget"
    stop_set = set(stop_tokens)
    vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 0) or model.config.vocab_size)
    for block in range(settings["generation_canvases"]):
        if chunks and time.perf_counter() - started >= settings.get("generation_seconds", 120):
            reason = "time_budget"
            break
        chunk = sample_response(
            model,
            history,
            steps=settings["sample_steps"],
            canvas_size=settings["canvas"],
            max_canvases=1,
            pad_id=pad_id,
            stop_tokens=stop_tokens,
            vocab_size=vocab_size,
            sampler=settings.get("sampler", "ddim"),
            temperature=settings.get("sample_temperature", 0.7),
            protect_stop_time=settings.get("protect_stop_time", 0.75),
            mask_id=settings.get("mask_id"),
            remask_eta=settings.get("remask_eta", 0.08),
        )
        chunks.append(chunk)
        print(
            "GENERATION_BLOCK",
            block + 1,
            "tokens",
            sum(len(c) for c in chunks),
            "seconds",
            round(time.perf_counter() - started, 2),
            flush=True,
        )
        if len(chunk) and int(chunk[-1]) in stop_set:
            reason = "stop_token"
            break
        history = torch.cat([history, chunk])
    return torch.cat(chunks), reason


@torch.inference_mode()
def evaluate(model, tokenizer, splits, plan, settings, out, step, generate, *, pad_id: int, eos_id: int, stop_tokens: tuple[int, ...]):
    was_training = model.training
    model.eval()
    canvas = settings["canvas"]
    device = next(model.parameters()).device
    vocab = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 0) or model.config.vocab_size)
    report = {"step": step, "recovery": [], "samples": [], "generation_requested": generate}
    try:
        for item in plan["recovery"]:
            row = splits[item["split"]][item["row"]]
            prompt, target = tensor_row(row, device)
            begin = item["block"] * canvas
            body = target[begin : begin + canvas]
            clean = target.new_full((canvas,), eos_id)
            clean[: len(body)] = body
            history = torch.cat([prompt, target[:begin]])
            for noise in (0.1, 0.5, 1.0):
                torch.manual_seed(90000 + item["row"] * 100 + item["block"] * 10 + int(noise * 10))
                changed = torch.rand(canvas, device=device) < noise
                if settings.get("corruption_mode") == "mask":
                    corrupt_fill = torch.full((canvas,), int(settings["mask_id"]), dtype=torch.long, device=device)
                else:
                    corrupt_fill = torch.randint(0, vocab, (canvas,), device=device)
                noisy = torch.where(changed, corrupt_fill, clean)
                logits = denoise_logits(model, history, noisy, pad_id=pad_id)[0]
                ce = F.cross_entropy(logits.float(), clean, reduction="none")
                correct = logits.argmax(-1) == clean
                record = dict(item, bucket=row["bucket"], noise=noise)
                for region, region_slice in (("body", slice(None, len(body))), ("tail", slice(len(body), None))):
                    moved, right = changed[region_slice], correct[region_slice]
                    record.update(
                        {
                            f"{region}_count": len(right),
                            f"{region}_changed": int(moved.sum()),
                            f"{region}_restored": int((moved & right).sum()),
                            f"{region}_kept": int((~moved).sum()),
                            f"{region}_preserved": int((~moved & right).sum()),
                            f"{region}_ce_sum": float(ce[region_slice].sum()),
                        }
                    )
                report["recovery"].append(record)
        if generate:
            for item in plan["generation"]:
                row = splits[item["split"]][item["row"]]
                prompt = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                torch.manual_seed(item["seed"])
                print("GENERATE_START", step, item, flush=True)
                output, reason = sample_pilot_response(model, prompt, settings, pad_id=pad_id, stop_tokens=stop_tokens)
                raw_text = tokenizer.decode(output.tolist(), skip_special_tokens=False)
                record = dict(
                    item,
                    prompt=row["prompt"],
                    reference=row["response"],
                    text=raw_text,
                    output_tokens=len(output),
                    target_tokens=len(row["response_ids"]),
                    stop_reason=reason,
                    has_think_close="</think>" in raw_text,
                )
                report["samples"].append(record)
                with (out / "samples.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(dict(step=step, **record), ensure_ascii=False) + "\n")
                print("ANSWER", step, item["row"], reason, repr(raw_text[:280]), flush=True)
        summary = []
        for split in sorted({r["split"] for r in report["recovery"]}):
            for noise in (0.1, 0.5, 1.0):
                group = [r for r in report["recovery"] if r["split"] == split and r["noise"] == noise]
                agg = {"split": split, "noise": noise, "selected_canvases": len(group)}
                for region in ("body", "tail"):
                    cnt = sum(r[f"{region}_count"] for r in group)
                    chg = sum(r[f"{region}_changed"] for r in group)
                    kept = sum(r[f"{region}_kept"] for r in group)
                    agg[region] = {
                        "tokens": cnt,
                        "ce": sum(r[f"{region}_ce_sum"] for r in group) / cnt if cnt else None,
                        "restored": sum(r[f"{region}_restored"] for r in group) / chg if chg else None,
                        "preserved": sum(r[f"{region}_preserved"] for r in group) / kept if kept else None,
                    }
                summary.append(agg)
        report["aggregate"] = summary
        write_json(out / f"eval-{step:04d}.json", report)
        for agg in summary:
            print("RECOVERY", step, json.dumps(agg), flush=True)
        return report
    finally:
        model.train(was_training)


def tiny_setup():
    tokenizer = ByteTokenizer()
    tokenizer.eos_token_id = 1
    config = Qwen3Config(
        vocab_size=259,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    splits = {}
    for split in ("train", "valid"):
        splits[split] = [
            {
                "prompt": f"{split}{i}",
                "response": "x" * (i * 16 + 1),
                "bucket": i,
                "prompt_ids": tokenizer.encode(f"{split}{i}"),
                "response_ids": tokenizer.encode("x" * (i * 16 + 1)) + [1],
            }
            for i in range(4)
        ]
    return model, tokenizer, splits


def apply_lora_if_requested(model, lora_rank: int):
    if lora_rank <= 0:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    # Keep the newly initialized SelfConditioner fully trainable alongside LoRA adapters.
    cond = _get_conditioner(model)
    for param in cond.parameters():
        param.requires_grad_(True)
    return model


def schedule_index(micro_step: int, size: int, seed: int = 42) -> int:
    epoch, offset = divmod(micro_step, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def compute_lr(
    step: int,
    *,
    max_steps: int,
    peak_lr: float,
    warmup_steps: int = 50,
    min_lr_ratio: float = 0.1,
    scheduler: str = "cosine",
    start_step: int = 0,
) -> float:
    """Linear warmup (for fresh runs) followed by cosine annealing or constant schedule."""
    if start_step == 0 and warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * (step / warmup_steps)
    if scheduler == "constant":
        return peak_lr
    decay_start = warmup_steps if start_step == 0 else start_step
    denom = max(max_steps - decay_start, 1)
    progress = min(max((step - decay_start) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_conversion_pilot_v1")
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/qwen3_8b_conversion_v1"))
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1, help="Rows packed per GPU forward pass")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=12288, help="OOM guard on padded micro-batch tokens")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing to trade VRAM for speed")
    parser.add_argument("--eval-every", type=int, default=256)
    parser.add_argument("--generation-every", type=int, default=256)
    parser.add_argument("--generation-canvases", type=int, default=18)
    parser.add_argument("--generation-seconds", type=int, default=120)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--sampler", choices=["ddim", "mask", "remdm", "rcd", "entropy"], default="ddim")
    parser.add_argument("--residual-mode", choices=["standard", "rcd"], default="standard")
    parser.add_argument("--remask-eta", type=float, default=0.08)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--corruption-mode", choices=["uniform", "mask"], default="uniform")
    parser.add_argument(
        "--loss-mode",
        choices=["canvas-mean", "corrupted-only", "corrupted-only-weighted", "body-tail-balanced"],
        default="canvas-mean",
    )
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--protect-stop-time", type=float, default=0.75)
    parser.add_argument("--lora-rank", type=int, default=64, help="0 for full-parameter training; >0 for all-linear LoRA")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--optimizer-mode", choices=["auto", "fp32-master", "gpu-adamw"], default="auto")
    parser.add_argument("--training-mode", choices=["single-block", "all-blocks"], default="single-block")
    parser.add_argument("--max-train-blocks", type=int, default=6, help="Max contiguous blocks per row in all-blocks mode")
    parser.add_argument("--ar-teacher", type=str, default=None, help="Frozen AR teacher model for top-K soft-logit KL distillation")
    parser.add_argument("--kl-weight", type=float, default=0.0, help="Weight on AR teacher forward KL loss (body tokens only)")
    parser.add_argument("--kl-topk", type=int, default=64, help="Top-K vocabulary entries kept from AR teacher distribution")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Temperature for AR teacher KL distillation")
    parser.add_argument("--data-offset", type=int, default=0, help="Row schedule offset to skip previously trained rows")
    parser.add_argument("--init-self-conditioner", type=Path, default=None, help="Preload unwrapped SelfConditioner weights before LoRA")
    parser.add_argument("--cudnn-sdp", action="store_true", help="Prefer CUDNN SDPA over slower mem-efficient SDPA (Blackwell)")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.ar_teacher and args.kl_weight > 0.0 and args.training_mode != "all-blocks":
        raise ValueError("--ar-teacher with --kl-weight > 0 requires --training-mode all-blocks")
    fresh_destination(args.out)
    hardware = preflight(args.tiny, args.out)
    print("PREFLIGHT", hardware, flush=True)

    random.seed(42)
    torch.manual_seed(42)
    if args.tiny:
        model, tokenizer, splits = tiny_setup()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        splits = load_qwen3_splits(args.data, tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()

    ar_teacher = None
    if args.ar_teacher and args.kl_weight > 0.0 and not args.eval_only:
        if args.tiny:
            ar_teacher, _, _ = tiny_setup()
            ar_teacher = ar_teacher.to(next(model.parameters()).device).eval()
        else:
            ar_teacher = AutoModelForCausalLM.from_pretrained(args.ar_teacher, dtype=torch.bfloat16).cuda().eval()
        for param in ar_teacher.parameters():
            param.requires_grad_(False)
        print("AR_TEACHER_LOADED", {"model": args.ar_teacher, "kl_weight": args.kl_weight, "kl_topk": args.kl_topk, "kl_temp": args.kl_temp}, flush=True)

    pad_id, eos_id, stop_tokens = resolve_special_ids(tokenizer)
    vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 0) or model.config.vocab_size)
    mask_id = (
        int(args.mask_id)
        if args.mask_id is not None
        else resolve_mask_id(tokenizer, vocab_size, pad_id=pad_id, eos_id=eos_id)
    )
    prepare_conversion_training(model)
    if args.init_self_conditioner and not args.resume:
        _get_conditioner(model).load_state_dict(load_file(str(args.init_self_conditioner)), strict=True)
        print("INIT_SELF_CONDITIONER_LOADED", str(args.init_self_conditioner), flush=True)
    model = apply_lora_if_requested(model, args.lora_rank)
    if hasattr(model, "enable_input_require_grads") and args.lora_rank > 0:
        model.enable_input_require_grads()
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    opt_mode = args.optimizer_mode
    if opt_mode == "auto":
        opt_mode = "gpu-adamw" if args.lora_rank > 0 else "fp32-master"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt_mode == "fp32-master":
        from llamafactory.train.diffusion.precision import MasterAdamW, require_master_memory

        if not args.tiny:
            print("MASTER_MEMORY", require_master_memory(model), flush=True)
        optimizer = MasterAdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4, foreach=False)

    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("out", "resume", "data", "keep_checkpoints", "eval_only")
    }
    settings.update(
        canvas=16 if args.tiny else 256,
        capacity=18,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
        stop_tokens=list(stop_tokens),
        resolved_optimizer=opt_mode,
        trainable_params=sum(p.numel() for p in trainable_params),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    if args.eval_only and not args.resume:
        raise RuntimeError("--eval-only requires --resume <checkpoint>")
    step = 0
    if args.resume:
        if not (args.resume / "COMPLETE").is_file():
            raise RuntimeError("Incomplete checkpoint")
        restored = torch.load(args.resume / "diagnostic_state.pt", map_location="cpu", weights_only=False)
        if args.eval_only or args.residual_mode == "rcd":
            compat_keys = ("model", "lora_rank", "pad_id", "eos_id", "mask_id", "corruption_mode")
            if any(restored["settings"].get(k) != settings.get(k) for k in compat_keys):
                raise RuntimeError("Checkpoint model/tokenizer settings differ; cannot load")
        elif restored["settings"] != settings:
            compat_keys = ("model", "lora_rank", "pad_id", "eos_id", "sampler")
            if any(restored["settings"].get(k) != settings.get(k) for k in compat_keys):
                raise RuntimeError("Checkpoint settings/code differ; no model loaded")
        if args.lora_rank > 0:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as _load_sf

            adapter_file = args.resume / "adapter_model.safetensors"
            set_peft_model_state_dict(model, _load_sf(str(adapter_file)))
        else:
            load_model_weights(model, args.resume)
        _get_conditioner(model).load_state_dict(load_file(str(args.resume / "self_conditioner.safetensors")), strict=True)
        step = restored["global_step"]
        if not args.eval_only:
            optimizer.load_state_dict(restored["optimizer"])
            torch.set_rng_state(restored["cpu_rng"])
            if restored["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
            random.setstate(restored["python_rng"])

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "settings.json", settings)
    plan = fixed_plan(splits, settings["canvas"])
    write_json(args.out / "eval_plan.json", plan)
    summary = {"status": "running", "global_step": step, "max_steps": args.max_steps}
    write_json(args.out / "summary.json", summary)

    print("QWEN3_PILOT_CONFIG", {k: settings[k] for k in ("model", "lora_rank", "resolved_optimizer", "trainable_params", "pad_id", "eos_id")}, flush=True)

    if args.eval_only:
        with preserve_rng():
            evaluate(model, tokenizer, splits, plan, settings, args.out, step, True, pad_id=pad_id, eos_id=eos_id, stop_tokens=stop_tokens)
        summary.update(status="eval_only_complete", global_step=step)
        write_json(args.out / "summary.json", summary)
        print("QWEN3_PILOT_EVAL_DONE", args.out, flush=True)
        return

    if not args.resume:
        with preserve_rng():
            evaluate(model, tokenizer, splits, plan, settings, args.out, 0, True, pad_id=pad_id, eos_id=eos_id, stop_tokens=stop_tokens)

    device = next(model.parameters()).device
    block_counts = Counter()
    resume_start = step
    with (args.out / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < args.max_steps:
            started = time.perf_counter()
            step += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = compute_lr(
                step,
                max_steps=args.max_steps,
                peak_lr=settings["lr"],
                warmup_steps=args.warmup_steps,
                min_lr_ratio=args.min_lr_ratio,
                scheduler=args.lr_scheduler,
                start_step=resume_start,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            records, loss_sum = [], 0.0
            ce_loss_sum, kl_loss_sum = 0.0, 0.0
            trained_blocks_sum, trained_tokens_sum = 0, 0
            effective_batch = max(1, args.grad_accum * args.micro_batch)
            step_items = []
            canvas = settings["canvas"]
            for offset in range(effective_batch):
                micro = (step - 1) * effective_batch + offset
                index = schedule_index(micro + args.data_offset, len(splits["train"]))
                prompt, target = tensor_row(splits["train"][index], device)
                if args.training_mode == "all-blocks":
                    n_active = min(max(1, (target.numel() + canvas - 1) // canvas), 18)
                    n_sel = min(n_active, max(1, args.max_train_blocks))
                    est_len = int(prompt.numel() + (n_active - 1) * canvas + n_sel * canvas)
                else:
                    est_len = int(prompt.numel() + min(target.numel(), canvas * 18) // 2 + canvas)
                step_items.append((micro, index, prompt, target, est_len))
            if args.micro_batch > 1:
                step_items.sort(key=lambda item: item[4])
            chunks: list[list[tuple]] = []
            current: list[tuple] = []
            for item in step_items:
                cand_max = max((c[4] for c in current), default=0)
                cand_max = max(cand_max, item[4])
                if current and (len(current) >= args.micro_batch or (len(current) + 1) * cand_max > args.max_batch_tokens):
                    chunks.append(current)
                    current = [item]
                else:
                    current.append(item)
            if current:
                chunks.append(current)

            for chunk in chunks:
                torch.manual_seed(420000 + chunk[0][0])
                if args.training_mode == "all-blocks":
                    loss, parts = diffusion_all_blocks_step(
                        model,
                        [c[2] for c in chunk],
                        [c[3] for c in chunk],
                        canvas_size=canvas,
                        num_canvases=18,
                        pad_id=pad_id,
                        eos_id=eos_id,
                        time_min=args.time_min,
                        time_max=args.time_max,
                        diagnostics=True,
                        loss_mode=args.loss_mode,
                        corruption_mode=args.corruption_mode,
                        mask_id=mask_id,
                        max_train_blocks=args.max_train_blocks,
                        ar_teacher=ar_teacher,
                        kl_weight=args.kl_weight,
                        kl_topk=args.kl_topk,
                        kl_temp=args.kl_temp,
                        residual_mode=args.residual_mode,
                    )
                    trained_blocks_sum += int(parts.get("trained_blocks", 0))
                    trained_tokens_sum += int(parts.get("trained_tokens", 0))
                else:
                    loss, parts = diffusion_batch_step(
                        model,
                        [c[2] for c in chunk],
                        [c[3] for c in chunk],
                        canvas_size=canvas,
                        num_canvases=18,
                        pad_id=pad_id,
                        eos_id=eos_id,
                        time_min=args.time_min,
                        time_max=args.time_max,
                        diagnostics=True,
                        loss_mode=args.loss_mode,
                        corruption_mode=args.corruption_mode,
                        mask_id=mask_id,
                        residual_mode=args.residual_mode,
                    )
                    trained_blocks_sum += len(chunk)
                    trained_tokens_sum += len(chunk) * canvas
                weight = len(chunk) / effective_batch
                (loss * weight).backward()
                loss_sum += float(loss.detach()) * weight
                ce_loss_sum += float(parts.get("ce_loss", float(loss.detach()))) * weight
                kl_loss_sum += float(parts.get("kl_loss", 0.0)) * weight
                for item, diag in zip(chunk, parts["rows"]):
                    records.append(dict(row=item[1], **diag))
            norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            block_counts.update(str(r["selected_block"]) for r in records)
            record = {
                "step": step,
                "loss": loss_sum,
                "ce_loss": ce_loss_sum,
                "kl_loss": kl_loss_sum,
                "trained_blocks": trained_blocks_sum,
                "trained_tokens": trained_tokens_sum,
                "grad": float(norm),
                "lr": lr,
                "seconds": time.perf_counter() - started,
                "rows": records,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print("TRAIN", json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                optimizer.zero_grad(set_to_none=True)
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    args.out,
                    {"global_step": step, "settings": settings, "sampled_block_counts": dict(block_counts)},
                    keep=args.keep_checkpoints,
                )
                with preserve_rng():
                    evaluate(
                        model,
                        tokenizer,
                        splits,
                        plan,
                        settings,
                        args.out,
                        step,
                        step % args.generation_every == 0 or step == args.max_steps,
                        pad_id=pad_id,
                        eos_id=eos_id,
                        stop_tokens=stop_tokens,
                    )
                summary.update(global_step=step, sampled_block_counts=dict(block_counts))
                write_json(args.out / "summary.json", summary)

    summary.update(status="budget_complete_no_full_training", global_step=step)
    write_json(args.out / "summary.json", summary)
    print("QWEN3_PILOT_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
