#!/usr/bin/env python3
"""
Gemma-4-E2B Route 2 (II-A) Stage 2 Talker Trainer:
  100% Freezes Stage-1 Gemma-4-E2B Thinker (intelligence & tool-calling 100% untouched!),
  extracts exact multimodal last-layer 1536-dim continuous hidden_states on spoken assistant spans,
  and trains Gemma4SpeechGenerator (gemma_norm + input_proj + Gate-Fusion + 24-layer Qwen2 SpeechGenerator)
  with Read-Write (R=3, W=10) teacher-forcing on single-speaker 25Hz CosyVoice2 discrete speech units.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gemma4_omni_route2 import (  # noqa: E402
    Cosy2StreamingVocoder,
    Gemma4SpeechGenerator,
    TalkerStreamSession,
    ensure_gemma4_duplex_special_tokens,
)
from llamafactory.data.tool_utils import Gemma4ToolUtils  # noqa: E402


SR = 16000


def patch_trainable_tokens_layer() -> None:
    try:
        from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
    except ImportError:
        return
    if getattr(TrainableTokensLayer, "_gemma4_patched", False):
        return
    _orig_fa = TrainableTokensLayer.forward_adapters

    def _patched_fa(self, x: torch.Tensor, active_adapters, *a, **kw) -> torch.Tensor:
        if self.disable_adapters or not active_adapters or self.merged:
            return _orig_fa(self, x, active_adapters, *a, **kw)
        base_W = self.base_layer.weight.detach()
        if not torch.is_floating_point(x):
            res = F.embedding(x, base_W, padding_idx=getattr(self.base_layer, "padding_idx", None))
            for an in active_adapters:
                idx = torch.tensor(self.token_indices[an], dtype=torch.long, device=x.device)
                deltas = self.trainable_tokens_delta[an].to(device=res.device, dtype=res.dtype)
                res = res + torch.matmul(
                    (x.unsqueeze(-1) == idx).to(res.dtype),
                    deltas - base_W.index_select(0, idx).to(res.dtype),
                )
            scale = (
                self._get_embed_scale()
                if hasattr(self, "_get_embed_scale")
                else getattr(self.base_layer, "embed_scale", None)
            )
            if scale is not None:
                res = res * (scale.to(res.dtype) if isinstance(scale, torch.Tensor) else float(scale))
            return res
        res = F.linear(x, base_W, bias=getattr(self.base_layer, "bias", None))
        for an in active_adapters:
            idx = torch.tensor(self.token_indices[an], dtype=torch.long, device=x.device)
            deltas = self.trainable_tokens_delta[an].to(device=x.device, dtype=x.dtype)
            res = res.index_add(-1, idx, F.linear(x, deltas - base_W.index_select(0, idx).to(x.dtype)))
        return res

    TrainableTokensLayer.forward_adapters = _patched_fa
    TrainableTokensLayer._gemma4_patched = True


def count_audio_soft_tokens(mask_1d: torch.Tensor) -> int:
    t = int(mask_1d.shape[0])
    m = mask_1d
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        m = m[::2][:t_out]
        t = int(m.shape[0])
    return int(m.sum().item())


def encode_stage2_multimodal_item(
    item: Dict[str, Any],
    tokenizer,
    audio_fe: Gemma4AudioFeatureExtractor,
    audio_root: Path,
) -> Tuple[List[int], int, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Encode full duplex conversation and return (input_ids, last_gpt_start, last_gpt_end, feats, masks)."""
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    tools_raw = item.get("tools", "")
    if tools_raw:
        tool_decl = Gemma4ToolUtils.tool_formatter(json.loads(tools_raw))
        sys_str = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl}<turn|>\n"
    else:
        sys_str = "<|turn>system\nYou are a real-time full-duplex voice assistant.<turn|>\n"

    ids: List[int] = [tokenizer.bos_token_id] + enc(sys_str)
    feat_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    aud_paths = item.get("audios", [])
    aud_idx = 0

    convs = item["conversations"]
    last_start = 0
    last_end = 0

    for ti, turn in enumerate(convs):
        role = turn["from"]
        val = turn["value"]
        if isinstance(val, str) and val.startswith("<|mask_turn|>"):
            val = val[len("<|mask_turn|>") :]

        if role == "human":
            if val == "<audio>":
                ap = audio_root / aud_paths[aud_idx]
                aud_idx += 1
                wav, sr = sf.read(str(ap), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                f_out = audio_fe([wav], sampling_rate=SR, return_tensors="pt")
                f_1 = f_out["input_features"][0]
                m_1 = f_out["input_features_mask"][0]
                n_soft = count_audio_soft_tokens(m_1)
                feat_list.append(f_1)
                mask_list.append(m_1)
                ids.extend(
                    enc("<|turn>user\n")
                    + [tokenizer.boa_token_id]
                    + [tokenizer.audio_token_id] * n_soft
                    + [tokenizer.eoa_token_id]
                    + enc("<turn|>\n<|turn>model\n")
                )
            else:
                ids.extend(enc(f"<|turn>user\n{val}<turn|>\n<|turn>model\n"))
        elif role == "observation":
            ids.extend(enc(f"<|turn>tool\n{val}<turn|>\n<|turn>model\n"))
        elif role == "gpt":
            g_ids = enc(f"{val}<turn|>\n")
            if ti == len(convs) - 1:
                last_start = len(ids)
                # Exclude trailing <turn|>\n from spoken span extraction
                val_only_ids = enc(val)
                last_end = last_start + len(val_only_ids)
            ids.extend(g_ids)

    if feat_list:
        padded_feats = pad_sequence(feat_list, batch_first=True, padding_value=0.0)
        padded_masks = pad_sequence(mask_list, batch_first=True, padding_value=False)
    else:
        padded_feats, padded_masks = None, None

    return ids, last_start, last_end, padded_feats, padded_masks


@torch.inference_mode()
def extract_frozen_thinker_features(
    jsonl_path: Path,
    thinker,
    tokenizer,
    audio_fe: Gemma4AudioFeatureExtractor,
    audio_root: Path,
    device: str,
    dtype: torch.dtype,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_items is not None:
        rows = rows[:max_items]

    extracted: List[Dict[str, Any]] = []
    for item in rows:
        ids, s_idx, e_idx, feats, masks = encode_stage2_multimodal_item(item, tokenizer, audio_fe, audio_root)
        if e_idx <= s_idx or len(ids) > 2048:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        kw: Dict[str, Any] = {"input_ids": inp, "output_hidden_states": True, "use_cache": False, " return_dict": True}
        kw.pop(" return_dict", None)
        kw["return_dict"] = True
        if feats is not None:
            kw["input_features"] = feats.to(device=device, dtype=dtype)
            kw["input_features_mask"] = masks.to(device=device)

        out = thinker(**kw)
        last_h = out.hidden_states[-1][0]  # [L, 1536]
        target_ids = ids[s_idx:e_idx]
        # Position p-1 in causal LM hidden_states predicts token at position p!
        pred_h = last_h[s_idx - 1 : e_idx - 1].detach().cpu()
        piece_h, pieces = Gemma4SpeechGenerator.extract_spoken_gemma_spans(tokenizer, target_ids, pred_h)
        if piece_h is None or not pieces:
            continue
        extracted.append({
            "piece_hiddens": piece_h,
            "pieces": pieces,
            "speech_units": item["speech_units"],
            "spoken_text": item["spoken_text"],
        })
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--stage1_adapter", type=str, default="")
    parser.add_argument("--donor_talker", type=str, default="ICTNLP/LLaMA-Omni2-1.5B-Bilingual")
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data" / "gemma4_route2_stage2"))
    parser.add_argument("--audio_root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "outputs" / "gemma4_route2_stage2_talker"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr_bridge", type=float, default=5e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-5)
    parser.add_argument("--max_items", type=int, default=0, help=">0 for quick local smoke test")
    parser.add_argument("--skip_vocoder_preview", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    max_n = args.max_items if args.max_items > 0 else None

    t0 = time.time()
    print(f"[Stage 2] Loading frozen Thinker ({args.base_model}) on {device} ({dtype})...")
    patch_trainable_tokens_layer()
    tok_src = args.stage1_adapter if args.stage1_adapter else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    ensure_gemma4_duplex_special_tokens(tokenizer)
    audio_fe = Gemma4AudioFeatureExtractor.from_pretrained(args.base_model)

    thinker = Gemma4ForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype)
    if args.stage1_adapter:
        peft_m = PeftModel.from_pretrained(thinker, args.stage1_adapter)
        try:
            thinker = peft_m.merge_and_unload()
        except Exception:
            thinker = peft_m
    thinker = thinker.to(device).eval()
    thinker.requires_grad_(False)

    data_dir = Path(args.data_dir)
    audio_root = Path(args.audio_root)
    train_raw = extract_frozen_thinker_features(
        data_dir / "train.jsonl", thinker, tokenizer, audio_fe, audio_root, device, dtype, max_items=max_n
    )
    valid_raw = extract_frozen_thinker_features(
        data_dir / "valid.jsonl",
        thinker,
        tokenizer,
        audio_fe,
        audio_root,
        device,
        dtype,
        max_items=(min(8, max_n) if max_n else None),
    )
    print(f"[Stage 2] Extracted frozen hidden states: train={len(train_raw)}, valid={len(valid_raw)} ({time.time()-t0:.1f}s)")

    # Free Thinker VRAM completely before loading Talker!
    del thinker
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    talker = Gemma4SpeechGenerator.from_llama_omni2_pretrained(
        donor_repo_id=args.donor_talker,
        torch_dtype=dtype,
    ).to(device)

    def prepare_aligned(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in items:
            th, tids = talker.align_pieces_to_tts(it["piece_hiddens"], it["pieces"])
            if th is not None and tids is not None and len(it["speech_units"]) >= 5:
                out.append({
                    "gemma_hidden": th,
                    "tts_ids": tids.cpu(),
                    "speech_units": it["speech_units"],
                    "spoken_text": it["spoken_text"],
                    "piece_hiddens": it["piece_hiddens"],
                    "pieces": it["pieces"],
                })
        return out

    train_set = prepare_aligned(train_raw)
    valid_set = prepare_aligned(valid_raw)
    if not valid_set and train_set:
        valid_set = train_set[: min(4, len(train_set))]

    bridge_params = (
        list(talker.gemma_norm.parameters())
        + list(talker.input_proj.parameters())
        + list(talker.gate.parameters())
    )
    backbone_params = list(talker.model.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": bridge_params, "lr": args.lr_bridge, "weight_decay": 0.01},
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 0.01},
        ]
    )

    steps_per_epoch = max(1, math.ceil(len(train_set) / (args.batch_size * args.grad_accum)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.08))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @torch.inference_mode()
    def eval_valid() -> Tuple[float, float]:
        talker.eval()
        losses, gates = [], []
        for i in range(0, len(valid_set), args.batch_size):
            batch = valid_set[i : i + args.batch_size]
            bh = [x["gemma_hidden"].to(device=device, dtype=dtype) for x in batch]
            bt = [x["tts_ids"].to(device=device) for x in batch]
            bu = [x["speech_units"] for x in batch]
            l, gm = talker.forward_batch_loss(bh, bt, bu)
            losses.append(float(l.item()))
            gates.append(float(gm.item()))
        talker.train()
        return float(np.mean(losses)), float(np.mean(gates))

    best_val = float("inf")
    global_step = 0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 2] Training Talker for {args.epochs} epochs ({total_steps} opt steps)...")
    talker.train()
    for ep in range(1, args.epochs + 1):
        ep_t0 = time.time()
        random.shuffle(train_set)
        tr_losses: List[float] = []
        optimizer.zero_grad()

        num_batches = math.ceil(len(train_set) / args.batch_size)
        for bi in range(num_batches):
            batch = train_set[bi * args.batch_size : (bi + 1) * args.batch_size]
            bh = [x["gemma_hidden"].to(device=device, dtype=dtype) for x in batch]
            bt = [x["tts_ids"].to(device=device) for x in batch]
            bu = [x["speech_units"] for x in batch]

            loss, _ = talker.forward_batch_loss(bh, bt, bu)
            tr_losses.append(float(loss.item()))
            (loss / args.grad_accum).backward()

            if (bi + 1) % args.grad_accum == 0 or (bi + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(talker.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        val_loss, val_gate = eval_valid()
        saved_mark = ""
        if val_loss <= best_val:
            best_val = val_loss
            talker.save_checkpoint(str(out_dir))
            saved_mark = " [saved best]"
        print(
            f"  Epoch {ep}/{args.epochs} | train_loss={np.mean(tr_losses):.4f} | "
            f"val_loss={val_loss:.4f} | gate_mean={val_gate:.3f} | {time.time()-ep_t0:.1f}s{saved_mark}"
        )

    if not args.skip_vocoder_preview and valid_set:
        talker.eval()
        voc = Cosy2StreamingVocoder(device=("cuda" if device == "cuda" else "cpu"), use_zero_shot_prompt=True)
        for pi, sample in enumerate(valid_set[:2]):
            sess = TalkerStreamSession()
            th = sample["gemma_hidden"].to(device=device, dtype=dtype)
            tids = sample["tts_ids"].to(device=device)
            pred_units: List[int] = []
            for idx in range(0, len(tids), talker.read_len):
                fin = (idx + talker.read_len >= len(tids))
                fused, _ = talker.fuse_representations(th[idx : idx + talker.read_len], tids[idx : idx + talker.read_len])
                pred_units.extend(talker._decode_unit_chunk(fused, sess, is_finished=fin, temperature=0.8))
            wav = voc.synthesize_full(pred_units)
            wav_p = out_dir / f"val_preview_{pi}.wav"
            sf.write(str(wav_p), wav, 24000)
            print(f"[Preview {pi}] '{sample['spoken_text']}' -> {len(pred_units)} units ({len(wav)/24000:.2f}s) -> {wav_p}")

    print(f"✅ Stage 2 Talker saved to {out_dir}")


if __name__ == "__main__":
    main()
