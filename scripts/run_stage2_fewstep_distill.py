#!/usr/bin/env python3
"""Stage-2 few-step sampler distillation (e.g. 64-96 steps -> 16-24 steps) on top of a Stage-1 diffusion checkpoint."""
# ruff: noqa: E402

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_qwen3_conversion_pilot import (
    apply_lora_if_requested,
    compute_lr,
    fixed_plan,
    load_qwen3_splits,
    load_vision_splits,
    schedule_index,
    tensor_row,
    tiny_vl_setup,
)
from run_single_sample_controls import fresh_destination, preflight, write_json
from validate_diffusion_conversion import preserve_rng

from llamafactory.train.diffusion.adapter import get_diffusion_adapter, load_diffusion_backbone
from llamafactory.train.diffusion.loss import _get_conditioner, prepare_conversion_training
from llamafactory.train.diffusion.rl import few_step_distill_step, sample_response


@torch.inference_mode()
def evaluate_fewstep(
    model,
    tokenizer,
    splits,
    plan,
    settings,
    out: Path,
    step: int,
    *,
    pad_id: int,
    stop_tokens: tuple[int, ...],
):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    canvas = int(settings["canvas_size"])
    eval_steps_list = [int(s) for s in settings["eval_steps_list"]]
    report = {"step": step, "samples": []}
    try:
        for n_steps in eval_steps_list:
            total_tokens = 0
            total_secs = 0.0
            for item in plan["generation"]:
                row = splits[item["split"]][item["row"]]
                prompt = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
                mm_in = row.get("multimodal")
                torch.manual_seed(int(item["seed"]))
                t0 = time.perf_counter()
                output = sample_response(
                    model,
                    prompt,
                    max_canvases=int(settings["generation_canvases"]),
                    steps=n_steps,
                    canvas_size=canvas,
                    pad_id=pad_id,
                    stop_tokens=stop_tokens,
                    sampler="frontier",
                    temperature=float(settings["sample_temperature"]),
                    multimodal_input=mm_in,
                    use_kv_cache=True,
                    repetition_penalty=float(settings["repetition_penalty"]),
                    defer_adjacent_duplicates=True,
                )
                dt = max(time.perf_counter() - t0, 1e-6)
                n_tok = int(output.numel())
                total_tokens += n_tok
                total_secs += dt
                raw_text = tokenizer.decode(output.tolist(), skip_special_tokens=False)
                stop_set = set(stop_tokens)
                reason = "stop_token" if (n_tok > 0 and int(output[-1]) in stop_set) else "canvas_budget"
                rec = dict(
                    step=step,
                    eval_steps=n_steps,
                    split=item["split"],
                    row=item["row"],
                    tokens=n_tok,
                    seconds=round(dt, 3),
                    tok_per_s=round(n_tok / dt, 1),
                    stop_reason=reason,
                    text=raw_text,
                )
                report["samples"].append(rec)
                print(
                    f"FEWSTEP_EVAL step={step} S={n_steps} {item['split']}[{item['row']}] "
                    f"{n_tok}tok/{dt:.2f}s ({n_tok/dt:.1f} tok/s) {reason} {raw_text[:220]!r}",
                    flush=True,
                )
            avg_tps = total_tokens / max(total_secs, 1e-6)
            print(
                f"FEWSTEP_SUMMARY step={step} S={n_steps} total_tokens={total_tokens} "
                f"total_seconds={total_secs:.2f} avg_tok_per_s={avg_tps:.1f}",
                flush=True,
            )
        write_json(out / f"eval-{step:04d}.json", report)
        with (out / "samples.jsonl").open("a", encoding="utf-8") as log:
            for rec in report["samples"]:
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return report
    finally:
        model.train(was_training)


def save_stage2_checkpoint(
    model,
    tokenizer,
    optimizer,
    out: Path,
    state: dict,
    *,
    keep: int = 2,
    hf_repo: str | None = None,
):
    step = int(state["global_step"])
    staging = out / f".checkpoint-{step}.writing"
    target = out / f"checkpoint-{step}"
    if staging.exists() or target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    staging.mkdir(parents=True)
    model.save_pretrained(staging)
    # PEFT writes a README.md with `base_model: <local_path>` that HF Hub rejects on upload.
    readme_path = staging / "README.md"
    if readme_path.is_file():
        readme_path.unlink()
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(staging)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in _get_conditioner(model).state_dict().items()},
        str(staging / "self_conditioner.safetensors"),
    )
    payload = dict(
        state,
        optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        python_rng=random.getstate(),
    )
    torch.save(payload, staging / "stage2_state.pt")
    (staging / "COMPLETE").write_text(str(step))
    staging.rename(target)
    write_json(out / "latest_checkpoint.json", {"path": str(target)})
    print("STAGE2_CHECKPOINT_COMPLETE", target, flush=True)
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=hf_repo, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(target),
                repo_id=hf_repo,
                repo_type="model",
                path_in_repo=target.name,
                ignore_patterns=["README.md"],
                commit_message=f"Stage-2 few-step checkpoint step {step}",
            )
            print("HF_CHECKPOINT_UPLOADED", f"{hf_repo}/{target.name}", flush=True)
        except Exception as exc:
            print("HF_CHECKPOINT_UPLOAD_FAILED", f"{hf_repo}/{target.name}", repr(exc), flush=True)

    ckpts = sorted(
        [p for p in out.glob("checkpoint-*") if (p / "COMPLETE").is_file()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for prev in ckpts[:-keep]:
        shutil.rmtree(prev)


def export_merged_stage2(model, tokenizer, processor, out: Path, *, hf_repo: str | None = None):
    merged_dir = out / "merged_stage2"
    merged_dir.mkdir(parents=True, exist_ok=True)
    cond_state = {k: v.detach().cpu().contiguous() for k, v in _get_conditioner(model).state_dict().items()}
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    try:
        merged.save_pretrained(merged_dir, max_shard_size="2GB")
    except TypeError:
        merged.save_pretrained(merged_dir)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(merged_dir)
    elif hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(merged_dir)
    save_file(cond_state, str(merged_dir / "self_conditioner.safetensors"))
    (merged_dir / "COMPLETE").write_text("merged_stage2")
    print("STAGE2_MERGED_EXPORTED", merged_dir, flush=True)
    if hf_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=hf_repo, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(merged_dir),
                repo_id=hf_repo,
                repo_type="model",
                path_in_repo="merged_stage2",
                ignore_patterns=["README.md"],
                commit_message="Stage-2 merged few-step model",
            )
            print("HF_MERGED_UPLOADED", f"{hf_repo}/merged_stage2", flush=True)
        except Exception as exc:
            print("HF_MERGED_UPLOAD_FAILED", f"{hf_repo}/merged_stage2", repr(exc), flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-ckpt", type=str, default="Qwen/Qwen3-VL-4B-Thinking", help="Local Stage-1 checkpoint dir (or HF repo)")
    parser.add_argument("--processor-model", type=str, default="Qwen/Qwen3-VL-4B-Thinking", help="Base HF model for tokenizer/processor config")
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_conversion_pilot_v1")
    parser.add_argument("--vision-data", type=Path, default=None)
    parser.add_argument("--vision-ratio", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/qwen3_vl_4b_stage2_fewstep"))
    parser.add_argument("--hf-repo", type=str, default=None, help="Optional private HF repo to sync checkpoints and merged model")
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--student-steps", type=int, default=16)
    parser.add_argument("--teacher-substeps", type=int, default=4)
    parser.add_argument("--train-points", type=int, default=3)
    parser.add_argument("--loss-mode", choices=["canvas-mean", "tail-capped"], default="tail-capped")
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-topk", type=int, default=64)
    parser.add_argument("--kl-temp", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=16, help="Rows accumulated per optimizer step")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--eval-steps-list", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--generation-canvases", type=int, default=6)
    parser.add_argument("--sample-temperature", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--data-offset", type=int, default=200000, help="Offset into text split so Stage-2 sees fresh prompts")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--cudnn-sdp", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--export-merged", action="store_true", help="Merge LoRA weights into standalone BF16 model at completion")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from an earlier Stage-2 checkpoint")
    parser.add_argument("--tiny-vl", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    fresh_destination(args.out)
    hardware = preflight(args.tiny_vl, args.out)
    print("PREFLIGHT", hardware, flush=True)

    random.seed(42)
    torch.manual_seed(42)
    processor = None
    if args.tiny_vl:
        model, tokenizer, splits = tiny_vl_setup()
        adapter = get_diffusion_adapter(model)
        canvas_size = 16
    else:
        model, tokenizer, processor, adapter = load_diffusion_backbone(
            args.stage1_ckpt,
            dtype=torch.bfloat16,
            device="cuda",
            processor_name=args.processor_model,
        )
        splits = load_qwen3_splits(args.data, tokenizer)
        if args.vision_data is not None:
            v_splits = load_vision_splits(args.vision_data, processor)
            if "train" in v_splits:
                splits["vision_train"] = v_splits["train"]
            if "valid" in v_splits:
                splits["vision_valid"] = v_splits["valid"]
            print("VISION_DATA_LOADED", {k: len(v) for k, v in v_splits.items()}, flush=True)
        canvas_size = int(args.canvas_size)

    pad_id, eos_id, stop_tokens = adapter.resolve_special_ids(tokenizer)
    prepare_conversion_training(model)
    sc_path = Path(args.stage1_ckpt) / "self_conditioner.safetensors"
    if not args.tiny_vl and sc_path.is_file():
        _get_conditioner(model).load_state_dict(load_file(str(sc_path)), strict=True)
        print("STAGE1_SELF_CONDITIONER_LOADED", str(sc_path), flush=True)

    # Frozen Stage-1 SelfConditioner snapshot used alongside disable_adapter() for exact 4-to-1 teacher refinement.
    teacher_conditioner = copy.deepcopy(_get_conditioner(model)).eval()
    for p in teacher_conditioner.parameters():
        p.requires_grad_(False)

    frozen_vis = adapter.freeze_vision(model)
    if frozen_vis > 0:
        print("VISION_ENCODER_FROZEN", {"frozen_params": frozen_vis}, flush=True)

    model = apply_lora_if_requested(model, args.lora_rank, freeze_vision=True)
    if hasattr(model, "enable_input_require_grads") and args.lora_rank > 0:
        model.enable_input_require_grads()
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4, foreach=False)

    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("out", "resume", "data", "vision_data", "keep_checkpoints")
    }
    settings.update(
        canvas_size=canvas_size,
        pad_id=pad_id,
        eos_id=eos_id,
        stop_tokens=list(stop_tokens),
        adapter=adapter.name,
        trainable_params=sum(p.numel() for p in trainable_params),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    step = 0
    if args.resume:
        if not (args.resume / "COMPLETE").is_file():
            raise RuntimeError("Incomplete Stage-2 checkpoint")
        restored = torch.load(args.resume / "stage2_state.pt", map_location="cpu", weights_only=False)
        if args.lora_rank > 0:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, load_file(str(args.resume / "adapter_model.safetensors")))
        _get_conditioner(model).load_state_dict(load_file(str(args.resume / "self_conditioner.safetensors")), strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["cpu_rng"])
        if restored["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
        random.setstate(restored["python_rng"])
        step = int(restored["global_step"])
        print("STAGE2_RESUMED", {"from": str(args.resume), "step": step}, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "settings.json", settings)
    plan = fixed_plan(splits, canvas_size)
    # Keep eval crisp: 4 text valid rows + up to 2 vision_valid rows.
    plan["generation"] = [g for g in plan["generation"] if g["split"] in ("valid", "vision_valid")]
    write_json(args.out / "eval_plan.json", plan)
    print(
        "STAGE2_CONFIG",
        {k: settings[k] for k in ("stage1_ckpt", "lora_rank", "trainable_params", "canvas_size", "student_steps", "teacher_substeps", "kl_weight")},
        flush=True,
    )

    with preserve_rng():
        evaluate_fewstep(model, tokenizer, splits, plan, settings, args.out, step, pad_id=pad_id, stop_tokens=stop_tokens)

    vision_train = splits.get("vision_train")
    vision_ratio = min(max(float(args.vision_ratio), 0.0), 1.0) if (vision_train and len(vision_train) > 0) else 0.0
    device = next(model.parameters()).device

    with (args.out / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < args.max_steps:
            started = time.perf_counter()
            step += 1
            model.train()
            adapter.freeze_vision(model)
            optimizer.zero_grad(set_to_none=True)
            lr = compute_lr(
                step,
                max_steps=args.max_steps,
                peak_lr=float(args.lr),
                warmup_steps=int(args.warmup_steps),
                min_lr_ratio=float(args.min_lr_ratio),
                scheduler="cosine",
                start_step=0,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            grad_accum = max(1, int(args.grad_accum))
            n_vision_rows = int(round(grad_accum * vision_ratio)) if vision_ratio > 0.0 else 0
            loss_sum = 0.0
            ce_sum = 0.0
            kl_sum = 0.0
            for offset in range(grad_accum):
                micro = (step - 1) * grad_accum + offset
                if vision_train is not None and offset < n_vision_rows:
                    v_micro = (step - 1) * max(n_vision_rows, 1) + offset
                    idx = schedule_index(v_micro, len(vision_train), seed=2026)
                    row_obj = vision_train[idx]
                else:
                    idx = schedule_index(micro + int(args.data_offset), len(splits["train"]), seed=99)
                    row_obj = splits["train"][idx]
                prompt, target = tensor_row(row_obj, device)
                mm_in = row_obj.get("multimodal")
                torch.manual_seed(770000 + micro)
                _, metrics = few_step_distill_step(
                    model,
                    prompt,
                    target,
                    canvas_size=canvas_size,
                    num_canvases=18,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    student_steps=int(args.student_steps),
                    teacher_substeps=int(args.teacher_substeps),
                    train_points=int(args.train_points),
                    loss_mode=args.loss_mode,
                    kl_weight=float(args.kl_weight),
                    kl_topk=int(args.kl_topk),
                    kl_temp=float(args.kl_temp),
                    temperature=float(args.sample_temperature),
                    teacher_conditioner=teacher_conditioner,
                    multimodal_input=mm_in,
                    backward_per_point=True,
                    loss_scale=1.0 / grad_accum,
                )
                loss_sum += metrics["loss"] / grad_accum
                ce_sum += metrics["ce_loss"] / grad_accum
                kl_sum += metrics["kl_loss"] / grad_accum

            norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            rec = {
                "step": step,
                "loss": loss_sum,
                "ce_loss": ce_sum,
                "kl_loss": kl_sum,
                "grad": float(norm),
                "lr": lr,
                "seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step == 1 or step % 5 == 0:
                print("STAGE2_TRAIN", json.dumps(rec), flush=True)

            if step % args.eval_every == 0 or step == args.max_steps:
                optimizer.zero_grad(set_to_none=True)
                save_stage2_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    args.out,
                    {"global_step": step, "settings": settings},
                    keep=args.keep_checkpoints,
                    hf_repo=args.hf_repo,
                )
                with preserve_rng():
                    evaluate_fewstep(
                        model, tokenizer, splits, plan, settings, args.out, step, pad_id=pad_id, stop_tokens=stop_tokens
                    )

    if args.export_merged:
        export_merged_stage2(model, tokenizer, processor, args.out, hf_repo=args.hf_repo)
    print("STAGE2_DISTILL_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
