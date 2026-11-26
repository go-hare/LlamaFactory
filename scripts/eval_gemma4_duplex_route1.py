#!/usr/bin/env python3
"""
Colab / Linux GPU 与 macOS 通用：Gemma-4-E2B 路线一训练后闭环评测脚本
直接读取 `data/gemma4_duplex_route1/audios/*.wav` 中的真实 16kHz 语音、远场混响旁人干扰与真打断音频（无需 macOS say 命令），
验证训出的 ~15MB LoRA 适配器在以下 7 个核心关卡的表现与 CUDA/MPS 单帧毫秒级延迟：
1. 真实语音分片 1 直入 (零 ASR) -> 预期输出 <unused1> (继续倾听)
2. 句中喘气静音帧 (<unused0>)   -> 预期输出 <unused1> (防抢话)
3. 真实语音分片 2 直入 (零 ASR) -> 预期输出 <unused1> (继续倾听)
4. 说完静音触发异步工具         -> 预期输出 <unused2> + 口语垫话 + <|tool_call>call:...<tool_call|>
5. 后台工具运行等待帧           -> 预期输出 <unused3> (waiting_tool)
6. 工具结果返回 + 远场混响抗噪  -> 预期输出 <unused2> 播报 & 遇到 MIT RIR 远场旁人音频输出 <unused5> 忽略不断流
7. 近场真实语音插话打断         -> 预期输出 <unused6> (确认真打断)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
SR = 16000

DUPLEX_TOKENS = [f"<unused{i}>" for i in range(8)]
HUMAN_LABELS = {
    "<unused0>": "<unused0:纯静音输入>",
    "<unused1>": "<unused1:继续倾听(防抢话)>",
    "<unused2>": "<unused2:接话开口>",
    "<unused3>": "<unused3:等待后台工具>",
    "<unused4>": "<unused4:答完空闲>",
    "<unused5>": "<unused5:忽略远场旁人/杂音>",
    "<unused6>": "<unused6:确认近场真打断!>",
    "<unused7>": "<unused7:作废旧工具!>",
}


def count_audio_soft_tokens(mask_1d: torch.Tensor) -> int:
    t = int(mask_1d.shape[0])
    m = mask_1d
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        m = m[::2][:t_out]
        t = int(m.shape[0])
    return int(m.sum().item())


def encode_audio_ids(
    tokenizer: Any, fe: Gemma4AudioFeatureExtractor, wav_path: Path
) -> Tuple[List[int], torch.Tensor, torch.Tensor, float]:
    wav, sr_in = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr_in != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr_in, SR).squeeze(0).numpy()
    feat = fe([wav], sampling_rate=SR, return_tensors="pt")
    n_soft = count_audio_soft_tokens(feat["input_features_mask"][0])
    a_ids = [tokenizer.boa_token_id] + [tokenizer.audio_token_id] * n_soft + [tokenizer.eoa_token_id]
    return a_ids, feat["input_features"], feat["input_features_mask"], len(wav) / SR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument("--adapter-dir", type=str, required=True, help="训练产出的 LoRA 目录")
    p.add_argument("--audio-dir", type=str, default=str(REPO_ROOT / "data" / "gemma4_duplex_route1" / "audios"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    adapter_dir = Path(args.adapter_dir)
    tok_src = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    tokenizer.add_special_tokens({"additional_special_tokens": DUPLEX_TOKENS})
    fe = Gemma4AudioFeatureExtractor.from_pretrained(args.base_model)

    print(f"[评测] 加载底座 {args.base_model} + 适配器 {adapter_dir} (设备: {device}, 精度: {dtype})")
    base_model = Gemma4ForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir)).to(device)
    model.eval()

    audio_dir = Path(args.audio_dir)
    # 选取数据集音频目录中的真实北京天气分片 + MIT RIR 远场旁人混响 + 真打断切片
    wav_p1 = audio_dir / "nf_0_0_p1.wav"
    wav_p2 = audio_dir / "nf_0_0_p2.wav"
    wav_barge = audio_dir / "nf_0_0_barge.wav"
    far_candidates = sorted(audio_dir.glob("farfield_bystander_*.wav")) or sorted(audio_dir.glob("audioset_noise_*.wav"))
    wav_far = far_candidates[0] if far_candidates else wav_p1

    def enc(s: str) -> List[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    sys_prompt = (
        "<|turn>system\n<|think|>You are a helpful assistant.\n"
        "<|tool>declaration:get_weather{description:<|\"|>执行 get_weather<|\"|>,"
        "parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
        "<turn|>\n"
    )

    ticks: List[Tuple[str, str | Path, str]] = [
        ("Tick 1 [真实语音片1: '帮我查一下']", wav_p1, "user"),
        ("Tick 2 [句中喘气静音帧]", "<unused0>", "user"),
        ("Tick 3 [真实语音片2: '明天北京的天气']", wav_p2, "user"),
        ("Tick 4 [说完静音 -> 预期垫话 + <|tool_call>]", "<unused0>", "user"),
        ("Tick 5 [后台等待帧 -> 预期 <unused3>]", "<unused0>", "user"),
        ("Tick 6 [工具结果回传 -> 预期开口播报]", '{"weather":"晴","temp":"20度"}', "tool"),
        ("Tick 7 [MIT RIR 远场旁人混响 -> 预期 <unused5> 忽略不断流]", wav_far, "user"),
        ("Tick 8 [近场大声真打断 -> 预期 <unused6>]", wav_barge, "user"),
    ]

    hist_ids: List[int] = [tokenizer.bos_token_id] + enc(sys_prompt)
    hist_feats: List[torch.Tensor] = []
    hist_masks: List[torch.Tensor] = []
    turn_end_id = enc("<turn|>")[0]

    print("=" * 76)
    for desc, item, role in ticks:
        hist_ids.extend(enc(f"<|turn>{role}\n"))
        if isinstance(item, Path) and item.exists():
            a_ids, f_t, m_t, dur_s = encode_audio_ids(tokenizer, fe, item)
            hist_ids.extend(a_ids)
            hist_feats.append(f_t[0])
            hist_masks.append(m_t[0])
            disp = f"<音频 {item.name}: {dur_s:.2f}s, {len(a_ids)-2}帧软Token>"
        else:
            hist_ids.extend(enc(str(item)))
            disp = str(item)
        hist_ids.extend(enc("<turn|>\n<|turn>model\n"))

        inp = torch.tensor([hist_ids], dtype=torch.long, device=device)
        g_kw: Dict[str, Any] = {
            "input_ids": inp,
            "attention_mask": torch.ones_like(inp),
            "max_new_tokens": 42,
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
        ms = (time.perf_counter() - t0) * 1000

        new_toks = out_g[0, inp.size(1) :].tolist()
        if new_toks and new_toks[-1] == turn_end_id:
            new_toks = new_toks[:-1]
        first_s = tokenizer.decode(new_toks[:1], skip_special_tokens=False) if new_toks else ""
        full_s = tokenizer.decode(new_toks, skip_special_tokens=False)

        print(f"\n▶ {desc}  ({ms:.1f} ms)")
        print(f"  输入 : {disp}")
        print(f"  状态 : {HUMAN_LABELS.get(first_s, first_s)}")
        print(f"  输出 : {full_s}")
        hist_ids.extend(new_toks + enc("<turn|>\n"))

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
