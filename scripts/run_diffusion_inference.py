#!/usr/bin/env python3
"""Inference & AR-vs-Diffusion speed benchmark for trained multimodal/text block-diffusion checkpoints."""
# ruff: noqa: E402

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llamafactory.train.diffusion.adapter import load_diffusion_backbone
from llamafactory.train.diffusion.loss import _get_conditioner, prepare_conversion_training
from llamafactory.train.diffusion.rl import sample_response


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True, help="Local checkpoint directory or HF repo path")
    parser.add_argument("--processor-model", type=str, default="Qwen/Qwen3-VL-4B-Thinking", help="Base HF model for tokenizer/processor")
    parser.add_argument("--prompt", type=str, default="Explain in 3 concise bullet points how block-diffusion language models work.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for multimodal VLM inference")
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--sampler", choices=["dynamic", "frontier", "entropy"], default="dynamic")
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, default=6)
    parser.add_argument("--steps", type=int, default=128, help="Max denoising steps per block (dynamic early-exits automatically)")
    parser.add_argument("--tau", type=float, default=0.75, help="Confidence threshold for dynamic parallel decoding")
    parser.add_argument("--window", type=int, default=32, help="Lookahead window for dynamic parallel decoding")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--benchmark-ar", action="store_true", help="Also run standard AR model.generate() baseline for speedup comparison")
    return parser.parse_args()


def build_inputs(prompt_text: str, image_path: str | None, tokenizer, processor, *, max_image_side: int, device: str):
    if image_path is not None and processor is not None:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_image_side, max_image_side))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[txt], images=[img], return_tensors="pt")
        mm = {"pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"]}
        return enc["input_ids"][0].to(device), mm, enc

    txt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
    )
    ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=False), dtype=torch.long, device=device)
    return ids, None, {"input_ids": ids[None]}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, tok, proc, adapter = load_diffusion_backbone(
        args.ckpt, dtype=dtype, device=device, processor_name=args.processor_model
    )
    prepare_conversion_training(model)
    sc_path = Path(args.ckpt) / "self_conditioner.safetensors"
    if sc_path.is_file():
        _get_conditioner(model).load_state_dict(load_file(str(sc_path)), strict=True)
    model.eval()

    pad_id, eos_id, stop_tokens = adapter.resolve_special_ids(tok)
    prompt_ids, mm_in, raw_enc = build_inputs(
        args.prompt, args.image, tok, proc, max_image_side=args.max_image_side, device=device
    )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = sample_response(
            model,
            prompt_ids,
            max_canvases=args.max_blocks,
            steps=args.steps,
            canvas_size=args.canvas_size,
            pad_id=pad_id,
            stop_tokens=stop_tokens,
            sampler=args.sampler,
            temperature=args.temperature,
            multimodal_input=mm_in,
            use_kv_cache=True,
            repetition_penalty=args.repetition_penalty,
            defer_adjacent_duplicates=True,
            confidence_threshold=args.tau,
            frontier_window=args.window,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    n_tok = int(out_ids.numel())
    text = tok.decode(out_ids.tolist(), skip_special_tokens=False)
    print(f"=== DIFFUSION ({args.sampler}, canvas={args.canvas_size}, tau={args.tau}) ===")
    print(f"tokens={n_tok}  seconds={dt:.2f}s  speed={n_tok / dt:.1f} tok/s")
    print(text)

    if args.benchmark_ar:
        gen_kwargs = {k: v.to(device) for k, v in raw_enc.items() if isinstance(v, torch.Tensor)}
        if device == "cuda":
            torch.cuda.synchronize()
        t_ar = time.perf_counter()
        with torch.inference_mode():
            ar_out = model.generate(
                **gen_kwargs,
                max_new_tokens=max(n_tok, 64),
                do_sample=True,
                temperature=max(args.temperature, 1e-4),
                eos_token_id=list(stop_tokens),
                pad_token_id=pad_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ar = max(time.perf_counter() - t_ar, 1e-6)
        ar_new = ar_out[0, prompt_ids.numel() :]
        ar_tok = int(ar_new.numel())
        print(f"\n=== AUTOREGRESSIVE BASELINE (model.generate) ===")
        print(f"tokens={ar_tok}  seconds={dt_ar:.2f}s  speed={ar_tok / dt_ar:.1f} tok/s")
        print(f"SPEEDUP: {(n_tok / dt) / max(ar_tok / dt_ar, 1e-6):.2f}x")
        print(tok.decode(ar_new.tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
