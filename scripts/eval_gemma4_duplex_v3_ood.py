#!/usr/bin/env python3
"""
Gemma-4-E2B Route-1 v3 10 维度严格零泄漏 OOD 生产级盲测基准脚本

评测 10 大维度（100% 未见说话人、未见 MIT RIR 房间、未见 AudioSet 噪音、未见自回声与犹豫卡顿）：
1. hold_listening (<unused1>)              : 真实语音流 + 句中喘气静音防抢话
2. hesitation_stutter_hold (<unused1>)     : 未见音色自然犹豫词 ("呃，那个") 防抢话
3. minds14_zero_asr_tool (4-way exact)     : 未见真实人类银行电话语音 4 选 1 零 ASR 调工具
4. cross_domain_tool_name (4-way exact)    : 未见音色跨领域语音 (天气/航班/家居/导航/点餐...) 4 选 1 调工具
5. cross_domain_param_hit (Chinese entity) : 跨领域语音直接从声波抽中文实体参数准确率
6. async_wait (<unused3>)                  : 后台异步工具运行等待态
7. self_echo_residual_reject (<unused5>)   : 助手自己扬声器 AEC 残留自回声不自打断
8. farfield_rir_reject (<unused5>)         : 未见人声 + 未见 MIT RIR 远场旁人抗噪
9. audioset_noise_reject (<unused5>)       : 未见 AudioSet 环境噪音抗噪
10. noisy_nearfield_bargein (<unused6>) & bargein_cancel_tool (<unused6><unused7>c1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from llamafactory.data.tool_utils import Gemma4ToolUtils  # noqa: E402

SR = 16000
DUPLEX_TOKENS = [f"<unused{i}>" for i in range(8)]


def count_audio_soft_tokens(mask_1d: torch.Tensor) -> int:
    t = int(mask_1d.shape[0])
    m = mask_1d
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        m = m[::2][:t_out]
        t = int(m.shape[0])
    return int(m.sum().item())


def encode_audio(
    tokenizer: Any, fe: Gemma4AudioFeatureExtractor, audio_path: Path
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    wav, sr_in = sf.read(str(audio_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr_in != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr_in, SR).squeeze(0).numpy()
    feat = fe([wav], sampling_rate=SR, return_tensors="pt")
    n_soft = count_audio_soft_tokens(feat["input_features_mask"][0])
    ids = [tokenizer.boa_token_id] + [tokenizer.audio_token_id] * n_soft + [tokenizer.eoa_token_id]
    return ids, feat["input_features"][0], feat["input_features_mask"][0]


def patch_trainable_tokens_layer() -> None:
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer

    if getattr(TrainableTokensLayer, "_gemma4_patched", False):
        return
    _orig_fa = TrainableTokensLayer.forward_adapters

    def _patched_fa(self, x: torch.Tensor, active_adapters, *a, **kw) -> torch.Tensor:
        if self.disable_adapters or not active_adapters or self.merged:
            return _orig_fa(self, x, active_adapters, *a, **kw)
        self._check_overlapping_tokens(active_adapters)
        base_W = self.base_layer.weight.detach()
        if not torch.is_floating_point(x):
            res = F.embedding(x, base_W, padding_idx=getattr(self.base_layer, "padding_idx", None))
            for an in active_adapters:
                idx = torch.tensor(self.token_indices[an], dtype=torch.long, device=x.device)
                deltas = self.trainable_tokens_delta[an].to(device=res.device, dtype=res.dtype)
                delta_diff = deltas - base_W.index_select(0, idx).to(res.dtype)
                res = res + torch.matmul((x.unsqueeze(-1) == idx).to(res.dtype), delta_diff)
            scale = self._get_embed_scale() if hasattr(self, "_get_embed_scale") else None
            if scale is None:
                scale = getattr(self.base_layer, "embed_scale", getattr(self.base_layer, "scalar_embed_scale", None))
            if scale is not None:
                res = res * (scale.to(res.dtype) if isinstance(scale, torch.Tensor) else float(scale))
            return res
        res = F.linear(x, base_W, bias=getattr(self.base_layer, "bias", None))
        for an in active_adapters:
            idx = torch.tensor(self.token_indices[an], dtype=torch.long, device=x.device)
            deltas = self.trainable_tokens_delta[an].to(device=x.device, dtype=x.dtype)
            delta_diff = deltas - base_W.index_select(0, idx).to(x.dtype)
            res = res.index_add(-1, idx, F.linear(x, delta_diff))
        return res

    TrainableTokensLayer.forward_adapters = _patched_fa
    TrainableTokensLayer._gemma4_patched = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument("--adapter-dir", type=str, required=True)
    p.add_argument("--data-root", type=str, default=str(REPO_ROOT / "data"))
    p.add_argument("--ood-manifest", type=str, default=str(REPO_ROOT / "data/gemma4_duplex_route1_v3/ood_eval.json"))
    p.add_argument("--num-minds", type=int, default=20)
    p.add_argument("--num-xdomain", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    data_root = Path(args.data_root)
    manifest = json.loads(Path(args.ood_manifest).read_text(encoding="utf-8"))

    adapter_dir = Path(args.adapter_dir)
    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.add_special_tokens({"additional_special_tokens": DUPLEX_TOKENS})
    fe = Gemma4AudioFeatureExtractor.from_pretrained(args.base_model)

    patch_trainable_tokens_layer()
    print(f"[v3 OOD 盲测] 加载底座 {args.base_model} + 适配器 {adapter_dir} ({device}, {dtype})")
    base_model = Gemma4ForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
    model.eval()

    def enc(s: str) -> List[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    turn_end_id = enc("<turn|>")[0]

    def run_turn(
        hist_ids: List[int],
        hist_feats: List[torch.Tensor],
        hist_masks: List[torch.Tensor],
        item: str | Path,
        role: str = "user",
        max_new: int = 48,
    ) -> Tuple[str, str]:
        hist_ids.extend(enc(f"<|turn>{role}\n"))
        if isinstance(item, Path):
            a_ids, f_t, m_t = encode_audio(tokenizer, fe, item)
            hist_ids.extend(a_ids)
            hist_feats.append(f_t)
            hist_masks.append(m_t)
        else:
            hist_ids.extend(enc(item))
        hist_ids.extend(enc("<turn|>\n<|turn>model\n"))

        inp = torch.tensor([hist_ids], dtype=torch.long, device=device)
        g_kw: Dict[str, Any] = {
            "input_ids": inp,
            "attention_mask": torch.ones_like(inp),
            "max_new_tokens": max_new,
            "eos_token_id": turn_end_id,
            "pad_token_id": 0,
            "do_sample": False,
        }
        if hist_feats:
            max_t = max(x.size(0) for x in hist_feats)
            pf = torch.zeros((len(hist_feats), max_t, 128), dtype=dtype, device=device)
            pm = torch.zeros((len(hist_masks), max_t), dtype=torch.bool, device=device)
            for idx_f, (ft, mt) in enumerate(zip(hist_feats, hist_masks)):
                pf[idx_f, : ft.size(0)] = ft.to(device, dtype)
                pm[idx_f, : mt.size(0)] = mt.to(device)
            g_kw["input_features"] = pf
            g_kw["input_features_mask"] = pm

        with torch.no_grad():
            out_g = model.generate(**g_kw)
        new_toks = out_g[0, inp.size(1) :].tolist()
        if new_toks and new_toks[-1] == turn_end_id:
            new_toks = new_toks[:-1]
        first_s = tokenizer.decode(new_toks[:1], skip_special_tokens=False) if new_toks else ""
        full_s = tokenizer.decode(new_toks, skip_special_tokens=False)
        hist_ids.extend(new_toks + enc("<turn|>\n"))
        return first_s, full_s

    stats: Dict[str, List[int]] = {
        "hold_listening (<unused1>)": [0, 0],
        "hesitation_stutter_hold (<unused1>)": [0, 0],
        "minds14_zero_asr_tool (4-way real)": [0, 0],
        "cross_domain_tool_name (4-way OOD)": [0, 0],
        "cross_domain_param_hit (ZH entity)": [0, 0],
        "async_wait (<unused3>)": [0, 0],
        "self_echo_residual_reject (<unused5>)": [0, 0],
        "farfield_rir_reject (<unused5>)": [0, 0],
        "audioset_noise_reject (<unused5>)": [0, 0],
        "noisy_nearfield_bargein (<unused6>)": [0, 0],
        "bargein_cancel_tool (<unused6><unused7>c1)": [0, 0],
    }

    # --- Part 1: Real human PolyAI/minds14 OOD banking calls ---
    m_cases = manifest["minds14_tool_cases"][: args.num_minds]
    print(f"▶ 评测 Part 1: {len(m_cases)} 组真实人类 minds14 OOD 电话语音...")
    for idx, c in enumerate(m_cases):
        tool_decl_str = Gemma4ToolUtils.tool_formatter(json.loads(c["tools"]))
        sys_prompt = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl_str}<turn|>\n"
        hist_ids: List[int] = [tokenizer.bos_token_id] + enc(sys_prompt)
        hist_feats: List[torch.Tensor] = []
        hist_masks: List[torch.Tensor] = []

        if c.get("hesit_path"):
            fh, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["hesit_path"], "user", max_new=6)
            stats["hesitation_stutter_hold (<unused1>)"][1] += 1
            if fh == "<unused1>":
                stats["hesitation_stutter_hold (<unused1>)"][0] += 1

        for pi, rel in enumerate(c["part_paths"]):
            f1, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / rel, "user", max_new=6)
            stats["hold_listening (<unused1>)"][1] += 1
            if f1 == "<unused1>":
                stats["hold_listening (<unused1>)"][0] += 1
            if pi == 0 and len(c["part_paths"]) >= 2:
                fp, _ = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=6)
                stats["hold_listening (<unused1>)"][1] += 1
                if fp == "<unused1>":
                    stats["hold_listening (<unused1>)"][0] += 1

        _, full2 = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=52)
        stats["minds14_zero_asr_tool (4-way real)"][1] += 1
        if f"call:{c['expected_tool']}" in full2:
            stats["minds14_zero_asr_tool (4-way real)"][0] += 1

        f3, _ = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=6)
        stats["async_wait (<unused3>)"][1] += 1
        if f3 == "<unused3>":
            stats["async_wait (<unused3>)"][0] += 1

        if idx % 2 == 0 and c.get("barge_path"):
            _, fc = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["barge_path"], "user", max_new=12)
            stats["bargein_cancel_tool (<unused6><unused7>c1)"][1] += 1
            if "<unused6>" in fc and "<unused7>" in fc:
                stats["bargein_cancel_tool (<unused6><unused7>c1)"][0] += 1
        else:
            run_turn(hist_ids, hist_feats, hist_masks, c["obs"], "tool", max_new=24)
            if c.get("echo_path"):
                fe_t, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["echo_path"], "user", max_new=20)
                stats["self_echo_residual_reject (<unused5>)"][1] += 1
                if fe_t == "<unused5>":
                    stats["self_echo_residual_reject (<unused5>)"][0] += 1
            if c.get("far_path"):
                ff_t, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["far_path"], "user", max_new=20)
                stats["farfield_rir_reject (<unused5>)"][1] += 1
                if ff_t == "<unused5>":
                    stats["farfield_rir_reject (<unused5>)"][0] += 1
            if c.get("noise_path"):
                fn_t, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["noise_path"], "user", max_new=20)
                stats["audioset_noise_reject (<unused5>)"][1] += 1
                if fn_t == "<unused5>":
                    stats["audioset_noise_reject (<unused5>)"][0] += 1
            if c.get("barge_path"):
                fb_t, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["barge_path"], "user", max_new=8)
                stats["noisy_nearfield_bargein (<unused6>)"][1] += 1
                if fb_t == "<unused6>":
                    stats["noisy_nearfield_bargein (<unused6>)"][0] += 1

    # --- Part 2: Cross-domain unseen-speaker audio tool-calling & Chinese entity extraction ---
    x_cases = manifest["cross_domain_tool_cases"][: args.num_xdomain]
    print(f"▶ 评测 Part 2: {len(x_cases)} 组未见音色跨领域零 ASR 抽参调工具...")
    for idx, c in enumerate(x_cases):
        tool_decl_str = Gemma4ToolUtils.tool_formatter(json.loads(c["tools"]))
        sys_prompt = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl_str}<turn|>\n"
        hist_ids = [tokenizer.bos_token_id] + enc(sys_prompt)
        hist_feats = []
        hist_masks = []
        for rel in c["part_paths"]:
            run_turn(hist_ids, hist_feats, hist_masks, data_root / rel, "user", max_new=6)
        _, full_x = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=56)

        stats["cross_domain_tool_name (4-way OOD)"][1] += 1
        name_ok = f"call:{c['expected_tool']}" in full_x
        if name_ok:
            stats["cross_domain_tool_name (4-way OOD)"][0] += 1

        exp_vals = [str(v) for v in c.get("expected_args", {}).values() if str(v)]
        stats["cross_domain_param_hit (ZH entity)"][1] += 1
        param_ok = name_ok and all(v in full_x for v in exp_vals)
        if param_ok:
            stats["cross_domain_param_hit (ZH entity)"][0] += 1

        if idx < 4:
            print(f"  [跨领域样例 {idx+1}] 原句={c['transcript']} | 工具命中={name_ok} 参数全中={param_ok} | 输出={full_x}")

    print("\n" + "=" * 76)
    print("📊 Route-1 v3 11 维度严格零泄漏 OOD 生产级盲测成绩单")
    print("=" * 76)
    for k, (ok, tot) in stats.items():
        acc = (100.0 * ok / tot) if tot else 0.0
        print(f"  {k:45s}: {ok:3d} / {tot:3d}  ({acc:5.1f}%)")
    print("=" * 76)


if __name__ == "__main__":
    main()
