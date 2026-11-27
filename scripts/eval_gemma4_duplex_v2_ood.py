#!/usr/bin/env python3
"""
Gemma-4-E2B Route-1 v2 严格零泄漏 OOD 真实语音盲测基准脚本

使用训练阶段 100% 未见过的：
- PolyAI/minds14 (zh-CN) 真实人类电话语音（4 选 1 银行工具，零 ASR 抽意图）
- AudioLLMs/aishell_1_zh_test 未见说话人 + 未见 MIT RIR 房间混响（远场旁人抗噪 <unused5>）
- agkphysics/AudioSet 未见环境音（环境抗噪 <unused5>）
- AudioLLMs/aishell_1_zh_test 未见说话人 + AudioSet 背景噪（嘈杂近场真打断 <unused6> & 作废旧工具 <unused6><unused7>c1）

输出每个关键状态机维度的真实 OOD 准确率与毫秒级延迟报告。
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
    p.add_argument("--ood-manifest", type=str, default=str(REPO_ROOT / "data/gemma4_duplex_route1_v2/ood_eval.json"))
    p.add_argument("--num-cases", type=int, default=20)
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
    print(f"[OOD 盲测] 加载底座 {args.base_model} + 适配器 {adapter_dir} ({device}, {dtype})")
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
        max_new: int = 44,
    ) -> Tuple[str, str, float, List[int]]:
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

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_g = model.generate(**g_kw)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0

        new_toks = out_g[0, inp.size(1) :].tolist()
        if new_toks and new_toks[-1] == turn_end_id:
            new_toks = new_toks[:-1]
        first_s = tokenizer.decode(new_toks[:1], skip_special_tokens=False) if new_toks else ""
        full_s = tokenizer.decode(new_toks, skip_special_tokens=False)
        hist_ids.extend(new_toks + enc("<turn|>\n"))
        return first_s, full_s, ms, new_toks

    cases = manifest["minds14_tool_cases"][: args.num_cases]
    stats: Dict[str, List[int]] = {
        "hold_listening (<unused1>)": [0, 0],
        "zero_asr_tool_start (<unused2>)": [0, 0],
        "zero_asr_tool_name_exact (4-way)": [0, 0],
        "async_wait (<unused3>)": [0, 0],
        "farfield_rir_reject (<unused5>)": [0, 0],
        "audioset_noise_reject (<unused5>)": [0, 0],
        "noisy_nearfield_bargein (<unused6>)": [0, 0],
        "bargein_cancel_tool (<unused6><unused7>c1)": [0, 0],
    }

    print(f"开始评测 {len(cases)} 组严格未见真实语音 OOD 闭环...")
    for idx, c in enumerate(cases):
        tool_decl_str = Gemma4ToolUtils.tool_formatter(json.loads(c["tools"]))
        sys_prompt = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl_str}<turn|>\n"
        hist_ids: List[int] = [tokenizer.bos_token_id] + enc(sys_prompt)
        hist_feats: List[torch.Tensor] = []
        hist_masks: List[torch.Tensor] = []

        # 1. 听未见过的真实人类语音切片 + 句中喘气静音 -> 期望 <unused1>
        for pi, rel in enumerate(c["part_paths"]):
            f1, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / rel, "user", max_new=6)
            stats["hold_listening (<unused1>)"][1] += 1
            if f1 == "<unused1>":
                stats["hold_listening (<unused1>)"][0] += 1
            if pi == 0 and len(c["part_paths"]) >= 2:
                f_p, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=6)
                stats["hold_listening (<unused1>)"][1] += 1
                if f_p == "<unused1>":
                    stats["hold_listening (<unused1>)"][0] += 1

        # 2. 说完静音 -> 期望 <unused2> + 零 ASR 精准命中 4 选 1 银行工具
        f2, full2, _, _ = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=48)
        stats["zero_asr_tool_start (<unused2>)"][1] += 1
        if f2 == "<unused2>":
            stats["zero_asr_tool_start (<unused2>)"][0] += 1
        stats["zero_asr_tool_name_exact (4-way)"][1] += 1
        hit_tool = f"call:{c['expected_tool']}" in full2
        if hit_tool:
            stats["zero_asr_tool_name_exact (4-way)"][0] += 1

        # 3. 后台工具等待帧 -> 期望 <unused3>
        f3, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, "<unused0>", "user", max_new=6)
        stats["async_wait (<unused3>)"][1] += 1
        if f3 == "<unused3>":
            stats["async_wait (<unused3>)"][0] += 1

        # 分支 A (偶数样本): 在工具等待中遭遇未见近场语音打断 -> 期望 <unused6><unused7>c1
        if idx % 2 == 0 and c.get("barge_path"):
            _, full_cancel, _, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["barge_path"], "user", max_new=12)
            stats["bargein_cancel_tool (<unused6><unused7>c1)"][1] += 1
            if "<unused6>" in full_cancel and "<unused7>" in full_cancel:
                stats["bargein_cancel_tool (<unused6><unused7>c1)"][0] += 1
        else:
            # 分支 B (奇数样本): 工具结果回传 -> 播报中测未见远场混响 (<unused5>) + 未见环境噪 (<unused5>) + 未见嘈杂近场打断 (<unused6>)
            run_turn(hist_ids, hist_feats, hist_masks, c["obs"], "tool", max_new=24)
            if c.get("far_path"):
                f_far, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["far_path"], "user", max_new=20)
                stats["farfield_rir_reject (<unused5>)"][1] += 1
                if f_far == "<unused5>":
                    stats["farfield_rir_reject (<unused5>)"][0] += 1
            if c.get("noise_path"):
                f_ns, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["noise_path"], "user", max_new=20)
                stats["audioset_noise_reject (<unused5>)"][1] += 1
                if f_ns == "<unused5>":
                    stats["audioset_noise_reject (<unused5>)"][0] += 1
            if c.get("barge_path"):
                f_bg, _, _, _ = run_turn(hist_ids, hist_feats, hist_masks, data_root / c["barge_path"], "user", max_new=8)
                stats["noisy_nearfield_bargein (<unused6>)"][1] += 1
                if f_bg == "<unused6>":
                    stats["noisy_nearfield_bargein (<unused6>)"][0] += 1

        if idx < 3:
            print(f"  [样例 {idx+1}] 预期工具={c['expected_tool']} | 命中={hit_tool} | 模型输出={full2}")

    print("\n" + "=" * 72)
    print("📊 Route-1 v2 严格零泄漏 OOD 真实语音盲测成绩单")
    print("=" * 72)
    for k, (ok, tot) in stats.items():
        acc = (100.0 * ok / tot) if tot else 0.0
        print(f"  {k:44s}: {ok:3d} / {tot:3d}  ({acc:5.1f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
