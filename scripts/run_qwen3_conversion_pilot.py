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
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_single_sample_controls import fresh_destination, load_model_weights, preflight, save_checkpoint, write_json
from validate_diffusion_conversion import ByteTokenizer, preserve_rng

from llamafactory.train.diffusion.adapter import get_diffusion_adapter, load_diffusion_backbone
from llamafactory.train.diffusion.loss import (
    DiffusionHistoryCache,
    _get_conditioner,
    denoise_logits,
    diffusion_all_blocks_step,
    diffusion_batch_step,
    prepare_conversion_training,
    supports_history_kv_cache,
)
from llamafactory.train.diffusion.rl import sample_canvas, sample_response


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
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    eos_list = [int(x) for x in raw_eos] if isinstance(raw_eos, (list, tuple)) else ([int(raw_eos)] if raw_eos is not None else [])
    raw_pad = getattr(tokenizer, "pad_token_id", None)
    pad_id = endoftext if isinstance(endoftext, int) else (int(raw_pad) if raw_pad is not None else 0)
    eos_id = im_end if isinstance(im_end, int) else (eos_list[0] if eos_list else 1)
    if eos_id == pad_id:
        eos_id = 1 if pad_id != 1 else 2
    stop_tokens = tuple(sorted({int(x) for x in [eos_id, *eos_list] if int(x) != pad_id}))
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
        bos = getattr(self.tokenizer, "bos_token", None)
        if isinstance(bos, str) and bos and not q_prompt.startswith(bos):
            q_prompt = bos + q_prompt
        p_ids = self.tokenizer.encode(q_prompt, add_special_tokens=False)[: self.prompt_cap]
        r_ids = self.tokenizer.encode(q_response, add_special_tokens=False)[: self.response_cap]
        if not p_ids or not r_ids:
            _, fallback_eos, _ = resolve_special_ids(self.tokenizer)
            fallback_bos = getattr(self.tokenizer, "bos_token_id", None)
            p_ids = p_ids or [int(fallback_bos) if fallback_bos is not None else 0]
            r_ids = r_ids or [fallback_eos]
        return {
            "prompt": q_prompt,
            "response": q_response,
            "prompt_ids": p_ids,
            "response_ids": r_ids,
            "bucket": int(raw.get("bucket", 0)),
        }


def load_qwen3_splits(
    directory: Path,
    tokenizer,
    *,
    prompt_cap: int = 4096,
    response_cap: int = 18 * 256,
    train_prompt_cap: int | None = None,
    train_response_cap: int | None = None,
):
    return {
        "valid": LazyQwen3Split(directory / "valid.jsonl", tokenizer, prompt_cap=prompt_cap, response_cap=response_cap),
        "train": LazyQwen3Split(
            directory / "train.jsonl",
            tokenizer,
            prompt_cap=train_prompt_cap or prompt_cap,
            response_cap=train_response_cap or response_cap,
        ),
    }


class LazyQwen3VLSplit:
    """Lazy multimodal JSONL split for Qwen3-VL (image + text prompt -> ChatML response)."""

    def __init__(
        self,
        path: Path,
        processor,
        *,
        prompt_cap: int = 4096,
        response_cap: int = 18 * 256,
        max_image_side: int = 768,
    ):
        self.path = path
        self.root_dir = path.parent
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.prompt_cap = prompt_cap
        self.response_cap = response_cap
        self.max_image_side = max_image_side
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

    def _open_image(self, ref: str):
        from PIL import Image

        p = Path(ref)
        if not p.is_absolute():
            p = self.root_dir / p
        img = Image.open(p).convert("RGB")
        if self.max_image_side > 0 and max(img.size) > self.max_image_side:
            img.thumbnail((self.max_image_side, self.max_image_side))
        return img

    def __getitem__(self, index: int) -> dict:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw = json.loads(handle.readline().decode("utf-8"))

        img_refs = raw.get("images")
        if img_refs is None and raw.get("image"):
            img_refs = [raw["image"]]
        img_refs = list(img_refs or [])
        pil_images = [self._open_image(r) for r in img_refs]

        user_text = str(raw.get("user") or raw.get("question") or raw.get("prompt") or "Describe this image.")
        system_text = str(raw.get("system") or "")
        messages: list[dict] = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        user_content = [{"type": "image", "image": img} for img in pil_images]
        user_content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": user_content})

        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        p_ids = encoded["input_ids"][0].tolist()
        if len(p_ids) > self.prompt_cap:
            # Never truncate inside vision placeholder tokens if avoidable; keep full prefix if images fit.
            p_ids = p_ids[: self.prompt_cap]

        resp_raw = str(raw.get("response") or raw.get("answer") or "")
        # Qwen3-VL-*-Thinking chat_template appends `<|im_start|>assistant\n<think>\n` to the prompt;
        # keep <think>...</think> balanced for both CoT and direct-answer multimodal rows.
        prompt_tail = self.tokenizer.decode(p_ids[-8:], skip_special_tokens=False)
        if prompt_tail.rstrip().endswith("<think>"):
            if resp_raw.startswith("<think>\n"):
                resp_raw = resp_raw[len("<think>\n") :]
            elif resp_raw.startswith("<think>"):
                resp_raw = resp_raw[len("<think>") :].lstrip("\n")
            elif "</think>" not in resp_raw:
                resp_raw = f"</think>\n{resp_raw}"
        if not resp_raw.endswith("<|im_end|>"):
            resp_raw = f"{resp_raw}<|im_end|>"
        r_ids = self.tokenizer.encode(resp_raw, add_special_tokens=False)[: self.response_cap]
        if not r_ids:
            _, fallback_eos, _ = resolve_special_ids(self.tokenizer)
            r_ids = [fallback_eos]

        mm = None
        if encoded.get("pixel_values") is not None and encoded.get("image_grid_thw") is not None:
            mm = {
                "pixel_values": encoded["pixel_values"],
                "image_grid_thw": encoded["image_grid_thw"],
            }
            if encoded.get("mm_token_type_ids") is not None:
                mm["mm_token_type_ids"] = encoded["mm_token_type_ids"][0][: len(p_ids)]

        return {
            "prompt": f"[IMAGE:{','.join(str(x) for x in img_refs)}] {user_text}",
            "response": resp_raw,
            "prompt_ids": p_ids,
            "response_ids": r_ids,
            "bucket": int(raw.get("bucket", 0)),
            "multimodal": mm,
        }


def load_vision_splits(
    directory: Path,
    processor,
    *,
    prompt_cap: int = 4096,
    response_cap: int = 18 * 256,
    train_response_cap: int | None = None,
):
    out = {}
    for split in ("valid", "train"):
        p = directory / f"{split}.jsonl"
        if p.is_file():
            r_cap = (train_response_cap or response_cap) if split == "train" else response_cap
            out[split] = LazyQwen3VLSplit(p, processor, prompt_cap=prompt_cap, response_cap=r_cap)
    return out


def fixed_plan(splits, canvas: int):
    plan = {"recovery": [], "generation": []}
    for split, rows in splits.items():
        if split.startswith("vision_"):
            limit = min(len(rows), 8)
            for idx in range(limit):
                blocks = (len(rows[idx]["response_ids"]) + canvas - 1) // canvas
                plan["recovery"].append({"split": split, "row": idx, "block": 0 if blocks <= 1 else (idx % blocks)})
            for idx in range(min(len(rows), 2)):
                plan["generation"].append({"split": split, "row": idx, "seed": 100})
            continue
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


def sample_pilot_response(
    model,
    prompt,
    settings,
    *,
    pad_id: int,
    stop_tokens: tuple[int, ...],
    multimodal_input: dict | None = None,
):
    started = time.perf_counter()
    chunks, history = [], prompt
    reason = "canvas_budget"
    stop_set = set(stop_tokens)
    vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 0) or model.config.vocab_size)
    use_kv = bool(settings.get("use_kv_cache", True)) and supports_history_kv_cache(model)
    rep_pen = float(settings.get("repetition_penalty", 1.0))
    defer_dup = bool(settings.get("defer_adjacent_dup", False))
    conf_tau = float(settings.get("confidence_threshold", 0.70))
    f_win = int(settings.get("frontier_window", 32))
    gen_canvas = int(settings.get("eval_canvas") or 0) or int(settings["canvas"])
    history_cache = None
    if use_kv:
        history_cache = DiffusionHistoryCache(model, pad_id=pad_id)
        history_cache.prefill(prompt, multimodal_input=multimodal_input)
    for block in range(settings["generation_canvases"]):
        if chunks and time.perf_counter() - started >= settings.get("generation_seconds", 120):
            reason = "time_budget"
            break
        if history_cache is not None:
            canvas, _ = sample_canvas(
                model,
                history,
                steps=settings["sample_steps"],
                canvas_size=gen_canvas,
                pad_id=pad_id,
                vocab_size=vocab_size,
                sampler=settings.get("sampler", "ddim"),
                temperature=settings.get("sample_temperature", 0.7),
                protect_stop_tokens=stop_tokens if settings.get("protect_stop_time", 0.75) > 0 else (),
                protect_stop_time=settings.get("protect_stop_time", 0.75),
                mask_id=settings.get("mask_id"),
                remask_eta=settings.get("remask_eta", 0.08),
                history_cache=history_cache,
                repetition_penalty=rep_pen,
                exempt_tokens=stop_tokens,
                defer_adjacent_duplicates=defer_dup,
                confidence_threshold=conf_tau,
                frontier_window=f_win,
            )
            row = canvas[0]
            hit_idx = next((idx for idx, tid in enumerate(row.tolist()) if tid in stop_set), None)
            chunk = row[: hit_idx + 1] if hit_idx is not None else row
        else:
            chunk = sample_response(
                model,
                history,
                steps=settings["sample_steps"],
                canvas_size=gen_canvas,
                max_canvases=1,
                pad_id=pad_id,
                stop_tokens=stop_tokens,
                vocab_size=vocab_size,
                sampler=settings.get("sampler", "ddim"),
                temperature=settings.get("sample_temperature", 0.7),
                protect_stop_time=settings.get("protect_stop_time", 0.75),
                mask_id=settings.get("mask_id"),
                remask_eta=settings.get("remask_eta", 0.08),
                multimodal_input=multimodal_input,
                repetition_penalty=rep_pen,
                defer_adjacent_duplicates=defer_dup,
                confidence_threshold=conf_tau,
                frontier_window=f_win,
            )
        chunks.append(chunk)
        tot_tok = sum(len(c) for c in chunks)
        elapsed = max(time.perf_counter() - started, 1e-6)
        print(
            "GENERATION_BLOCK",
            block + 1,
            "tokens",
            tot_tok,
            "seconds",
            round(elapsed, 2),
            "tok_per_s",
            round(tot_tok / elapsed, 1),
            flush=True,
        )
        if len(chunk) and int(chunk[-1]) in stop_set:
            reason = "stop_token"
            break
        if history_cache is not None:
            history_cache.prefill(chunk)
        else:
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
            mm_in = row.get("multimodal")
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
                logits = denoise_logits(model, history, noisy, pad_id=pad_id, multimodal_input=mm_in)[0]
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
                mm_in = row.get("multimodal")
                torch.manual_seed(item["seed"])
                print("GENERATE_START", step, item, flush=True)
                output, reason = sample_pilot_response(
                    model, prompt, settings, pad_id=pad_id, stop_tokens=stop_tokens, multimodal_input=mm_in
                )
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


def tiny_vl_setup():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    tokenizer = ByteTokenizer()
    tokenizer.eos_token_id = 1
    img_tok = 250
    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=259,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        tie_word_embeddings=True,
        image_token_id=img_tok,
        video_token_id=251,
        vision_start_token_id=248,
        vision_end_token_id=249,
    )
    model = Qwen3VLForConditionalGeneration(cfg).to(torch.bfloat16)
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
    # Synthetic multimodal split (1 image of grid [1, 2, 2] -> 1 merged vision token per image)
    patch_dim = 3 * 2 * 4 * 4  # in_channels * temporal_patch_size * patch_size^2
    vision_rows = []
    for i in range(4):
        p_ids = [248, img_tok, 249] + tokenizer.encode(f"img{i}")
        vision_rows.append(
            {
                "prompt": f"[IMAGE:synthetic_{i}] img{i}",
                "response": "v" * (i * 8 + 4),
                "bucket": i % 2,
                "prompt_ids": p_ids,
                "response_ids": tokenizer.encode("v" * (i * 8 + 4)) + [1],
                "multimodal": {
                    "pixel_values": torch.randn(4, patch_dim, dtype=torch.bfloat16),
                    "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
                },
            }
        )
    splits["vision_train"] = vision_rows
    splits["vision_valid"] = vision_rows[:2]
    return model, tokenizer, splits


def apply_lora_if_requested(model, lora_rank: int, *, freeze_vision: bool = True):
    if lora_rank <= 0:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    adapter = get_diffusion_adapter(model)
    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        target_modules=adapter.lora_target_modules(model, freeze_vision=freeze_vision),
    )
    try:
        lora_cfg = LoraConfig(**lora_kwargs, exclude_modules=r".*self_conditioner.*")
    except TypeError:
        lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    if freeze_vision:
        adapter.freeze_vision(model)
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
    resume_warmup_steps: int = 0,
) -> float:
    """Linear warmup (for fresh runs or fresh-optimizer resumes) followed by cosine annealing or constant schedule."""
    if start_step == 0 and warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * (step / warmup_steps)
    if start_step > 0 and resume_warmup_steps > 0 and (step - start_step) <= resume_warmup_steps:
        return peak_lr * ((step - start_step) / resume_warmup_steps)
    if scheduler == "constant":
        return peak_lr
    decay_start = warmup_steps if start_step == 0 else (start_step + resume_warmup_steps)
    denom = max(max_steps - decay_start, 1)
    progress = min(max((step - decay_start) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--processor-model", type=str, default=None, help="Optional base HF model for tokenizer/processor config")
    parser.add_argument("--data", type=Path, default=ROOT / "data/real_conversion_pilot_v1")
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/saves/qwen3_8b_conversion_v1"))
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=1, help="Rows packed per GPU forward pass")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=12288, help="OOM guard on padded micro-batch tokens")
    parser.add_argument("--train-prompt-cap", type=int, default=4096, help="Hard cap on text train prompt tokens (keeps valid untouched)")
    parser.add_argument("--train-response-cap", type=int, default=18 * 256, help="Hard cap on train response tokens (bounds worst-case O(L^2) attention)")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing to trade VRAM for speed")
    parser.add_argument("--eval-every", type=int, default=256)
    parser.add_argument("--generation-every", type=int, default=256)
    parser.add_argument("--generation-canvases", type=int, default=18)
    parser.add_argument("--generation-seconds", type=int, default=120)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--sampler", choices=["ddim", "mask", "remdm", "rcd", "entropy", "frontier", "dynamic"], default="ddim")
    parser.add_argument("--confidence-threshold", type=float, default=0.70, help="Raw T=1.0 confidence threshold for dynamic sampler")
    parser.add_argument("--frontier-window", type=int, default=32, help="Lookahead window for dynamic sampler")
    parser.add_argument("--residual-mode", choices=["standard", "rcd"], default="standard")
    parser.add_argument("--remask-eta", type=float, default=0.08)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--corruption-mode", choices=["uniform", "mask"], default="uniform")
    parser.add_argument(
        "--loss-mode",
        choices=["canvas-mean", "tail-capped", "corrupted-only", "corrupted-only-weighted", "body-tail-balanced"],
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
    parser.add_argument(
        "--optimizer-mode",
        choices=["auto", "fp32-master", "gpu-fp32-master", "gpu-adamw"],
        default="auto",
    )
    parser.add_argument("--training-mode", choices=["single-block", "all-blocks"], default="single-block")
    parser.add_argument("--max-train-blocks", type=int, default=6, help="Max contiguous blocks per row in all-blocks mode")
    parser.add_argument("--ar-teacher", type=str, default=None, help="Frozen AR teacher model for top-K soft-logit KL distillation")
    parser.add_argument("--kl-weight", type=float, default=0.0, help="Weight on AR teacher forward KL loss (body tokens only)")
    parser.add_argument("--kl-topk", type=int, default=64, help="Top-K vocabulary entries kept from AR teacher distribution")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Temperature for AR teacher KL distillation")
    parser.add_argument("--data-offset", type=int, default=0, help="Row schedule offset to skip previously trained rows")
    parser.add_argument("--init-self-conditioner", type=Path, default=None, help="Preload unwrapped SelfConditioner weights before LoRA")
    parser.add_argument("--cudnn-sdp", action="store_true", help="Prefer CUDNN SDPA over slower mem-efficient SDPA (Blackwell)")
    parser.add_argument("--vision-data", type=Path, default=None, help="Optional multimodal JSONL directory (train.jsonl/valid.jsonl) to mix with text data")
    parser.add_argument("--vision-ratio", type=float, default=0.25, help="Fraction of training rows sampled from --vision-data when provided")
    parser.add_argument("--unfreeze-vision", action="store_true", help="Train vision encoder instead of keeping it frozen")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable History KV Cache during generation evaluation")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Local neighbor repetition penalty during sampling (>1.0 dampens comma runs)")
    parser.add_argument("--defer-adjacent-dup", action="store_true", help="Defer same-step adjacent duplicate commits by 1 step to eliminate parallel stuttering")
    parser.add_argument("--eval-canvas", type=int, default=0, help="Optional smaller canvas size during generation eval (e.g. 64 or 128; 0 = training canvas)")
    parser.add_argument("--no-save-optimizer", action="store_true", help="Omit optimizer state from checkpoints (~9GB vs ~55GB; prevents Google Drive FUSE upload stalls)")
    parser.add_argument("--hf-repo", type=str, default=None, help="Optional private HF model repo to synchronously upload each checkpoint (e.g. user/qwen3-vl-4b-diff)")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--tiny-vl", action="store_true", help="Run tiny synthetic multimodal Qwen3-VL end-to-end check")
    return parser.parse_args()


def main():
    args = arguments()
    is_tiny = bool(args.tiny or args.tiny_vl)
    freeze_vision = not bool(args.unfreeze_vision)
    if args.cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.ar_teacher and args.kl_weight > 0.0 and args.training_mode != "all-blocks":
        raise ValueError("--ar-teacher with --kl-weight > 0 requires --training-mode all-blocks")
    fresh_destination(args.out)
    hardware = preflight(is_tiny, args.out)
    print("PREFLIGHT", hardware, flush=True)

    random.seed(42)
    torch.manual_seed(42)
    processor = None
    if args.tiny_vl:
        model, tokenizer, splits = tiny_vl_setup()
        adapter = get_diffusion_adapter(model)
    elif args.tiny:
        model, tokenizer, splits = tiny_setup()
        adapter = get_diffusion_adapter(model)
    else:
        model, tokenizer, processor, adapter = load_diffusion_backbone(
            args.model, dtype=torch.bfloat16, device="cuda", processor_name=args.processor_model
        )
        splits = load_qwen3_splits(
            args.data,
            tokenizer,
            train_prompt_cap=args.train_prompt_cap,
            train_response_cap=args.train_response_cap,
        )
        if args.vision_data is not None:
            if processor is None:
                raise RuntimeError(f"--vision-data requires a multimodal processor, but {args.model} loaded without one.")
            v_splits = load_vision_splits(args.vision_data, processor, train_response_cap=args.train_response_cap)
            if "train" in v_splits:
                splits["vision_train"] = v_splits["train"]
            if "valid" in v_splits:
                splits["vision_valid"] = v_splits["valid"]
            print("VISION_DATA_LOADED", {k: len(v) for k, v in v_splits.items()}, flush=True)

    ar_teacher = None
    if args.ar_teacher and args.kl_weight > 0.0 and not args.eval_only:
        if args.tiny_vl:
            ar_teacher, _, _ = tiny_vl_setup()
            ar_teacher = ar_teacher.to(next(model.parameters()).device).eval()
        elif args.tiny:
            ar_teacher, _, _ = tiny_setup()
            ar_teacher = ar_teacher.to(next(model.parameters()).device).eval()
        else:
            ar_teacher, _, _, _ = load_diffusion_backbone(args.ar_teacher, dtype=torch.bfloat16, device="cuda")
            ar_teacher = ar_teacher.eval()
        for param in ar_teacher.parameters():
            param.requires_grad_(False)
        print("AR_TEACHER_LOADED", {"model": args.ar_teacher, "kl_weight": args.kl_weight, "kl_topk": args.kl_topk, "kl_temp": args.kl_temp}, flush=True)

    pad_id, eos_id, stop_tokens = adapter.resolve_special_ids(tokenizer)
    vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size", 0) or model.config.vocab_size)
    mask_id = (
        int(args.mask_id)
        if args.mask_id is not None
        else resolve_mask_id(tokenizer, vocab_size, pad_id=pad_id, eos_id=eos_id)
    )
    prepare_conversion_training(model)
    if freeze_vision:
        frozen_vis = adapter.freeze_vision(model)
        if frozen_vis > 0:
            print("VISION_ENCODER_FROZEN", {"frozen_params": frozen_vis}, flush=True)
    if args.init_self_conditioner and not args.resume:
        _get_conditioner(model).load_state_dict(load_file(str(args.init_self_conditioner)), strict=True)
        print("INIT_SELF_CONDITIONER_LOADED", str(args.init_self_conditioner), flush=True)
    model = apply_lora_if_requested(model, args.lora_rank, freeze_vision=freeze_vision)
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

        if not is_tiny:
            print("MASTER_MEMORY", require_master_memory(model), flush=True)
        optimizer = MasterAdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4)
    elif opt_mode == "gpu-fp32-master":
        from llamafactory.train.diffusion.precision import GpuMasterAdamW

        optimizer = GpuMasterAdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4, foreach=False)

    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("out", "resume", "data", "vision_data", "keep_checkpoints", "eval_only")
    }
    settings.update(
        canvas=16 if is_tiny else 256,
        capacity=18,
        pad_id=pad_id,
        eos_id=eos_id,
        mask_id=mask_id,
        stop_tokens=list(stop_tokens),
        use_kv_cache=not bool(args.no_kv_cache),
        adapter=adapter.name,
        resolved_optimizer=opt_mode,
        trainable_params=sum(p.numel() for p in trainable_params),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    if args.eval_only and not args.resume:
        raise RuntimeError("--eval-only requires --resume <checkpoint>")
    step = 0
    resume_warmup_steps = 0
    if args.resume:
        if not (args.resume / "COMPLETE").is_file():
            raise RuntimeError("Incomplete checkpoint")
        restored = torch.load(args.resume / "diagnostic_state.pt", map_location="cpu", weights_only=False)
        if args.eval_only or args.residual_mode == "rcd":
            compat_keys = ("model", "lora_rank", "pad_id", "eos_id", "mask_id", "corruption_mode")
            if any(restored["settings"].get(k) != settings.get(k) for k in compat_keys):
                raise RuntimeError("Checkpoint model/tokenizer settings differ; cannot load")
        elif restored["settings"] != settings:
            compat_keys = ("model", "lora_rank", "pad_id", "eos_id")
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
            if restored.get("optimizer") is not None:
                optimizer.load_state_dict(restored["optimizer"])
            else:
                resume_warmup_steps = min(20, max(1, args.max_steps - step))
                print("RESUME_FRESH_OPTIMIZER", {"step": step, "resume_warmup_steps": resume_warmup_steps}, flush=True)
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

    vision_train = splits.get("vision_train")
    vision_ratio = min(max(float(args.vision_ratio), 0.0), 1.0) if (vision_train and len(vision_train) > 0) else 0.0
    device = next(model.parameters()).device
    block_counts = Counter()
    resume_start = step
    with (args.out / "train.jsonl").open("a", encoding="utf-8") as log:
        while step < args.max_steps:
            started = time.perf_counter()
            step += 1
            model.train()
            if freeze_vision:
                adapter.freeze_vision(model)
            optimizer.zero_grad(set_to_none=True)
            lr = compute_lr(
                step,
                max_steps=args.max_steps,
                peak_lr=settings["lr"],
                warmup_steps=args.warmup_steps,
                min_lr_ratio=args.min_lr_ratio,
                scheduler=args.lr_scheduler,
                start_step=resume_start,
                resume_warmup_steps=resume_warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            records, loss_sum = [], 0.0
            ce_loss_sum, kl_loss_sum = 0.0, 0.0
            trained_blocks_sum, trained_tokens_sum = 0, 0
            effective_batch = max(1, args.grad_accum * args.micro_batch)
            step_items = []
            canvas = settings["canvas"]
            n_vision_rows = int(round(effective_batch * vision_ratio)) if vision_ratio > 0.0 else 0
            if vision_ratio > 0.0 and n_vision_rows == 0:
                n_vision_rows = 1 if (step % max(1, int(round(1.0 / vision_ratio))) == 0) else 0
            for offset in range(effective_batch):
                micro = (step - 1) * effective_batch + offset
                use_vision_row = vision_train is not None and offset < n_vision_rows
                if use_vision_row:
                    v_micro = (step - 1) * max(n_vision_rows, 1) + offset
                    index = schedule_index(v_micro, len(vision_train), seed=1337)
                    row_obj = vision_train[index]
                    row_id = f"v:{index}"
                else:
                    index = schedule_index(micro + args.data_offset, len(splits["train"]))
                    row_obj = splits["train"][index]
                    row_id = index
                prompt, target = tensor_row(row_obj, device)
                mm_in = row_obj.get("multimodal")
                if args.training_mode == "all-blocks":
                    n_active = min(max(1, (target.numel() + canvas - 1) // canvas), 18)
                    n_sel = min(n_active, max(1, args.max_train_blocks))
                    est_len = int(prompt.numel() + (n_active - 1) * canvas + n_sel * canvas)
                else:
                    est_len = int(prompt.numel() + min(target.numel(), canvas * 18) // 2 + canvas)
                step_items.append((micro, row_id, prompt, target, est_len, mm_in))
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
                chunk_mm = [c[5] for c in chunk]
                mm_arg = chunk_mm if any(x is not None for x in chunk_mm) else None
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
                        multimodal_inputs=mm_arg,
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
                        multimodal_inputs=mm_arg,
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
            if step == 1 or step == resume_start + 1 or step % 10 == 0:
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
                    save_optimizer=not bool(args.no_save_optimizer),
                    hf_repo=args.hf_repo,
                    processor=processor,
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
