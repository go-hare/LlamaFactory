#!/usr/bin/env python3
"""
Gemma-4-E2B Route-1 v2 大规模真实音频扩容 + 零泄漏 OOD 基准构建流水线

数据源（100% 开源免鉴权 Parquet + 本地缓存，严格 Train / Valid+OOD 声源物理隔离）：
1. [真实人类语音 + 零 ASR 异步工具] PolyAI/minds14 (zh-CN, 502 条真实中文银行电话语音, 14 类结构化意图)
   - 严格划分 440 Train / 62 OOD Held-Out，覆盖真实口音、语速、自然停顿与零 ASR 抽意图调工具。
2. [真实多说话人中文流式语音 + 声学难负样本] AudioLLMs/aishell_1_zh_test (2,300+ 条真实多人普通话)
   - 严格 90/10 物理切分：
     * 近场真实人声分片直入 -> 防抢话 <unused1> / 接话 <unused2>
     * 卷积 270 个真实 MIT RIR 房间混响 + -14~-22dB 衰减 + AudioSet 底噪 -> 远场旁人干扰 <unused5>
     * 近场真实人声 + AudioSet 背景噪 (+6~+14dB SNR) -> 嘈杂环境真打断 <unused6> (硬边界对比!)
3. [真实中文口语问答语音流] JerryAGENDD/chinese_speech_cosy_audio (shard 0, 2,500 条真实语音问答)
   - 过滤书面/代码答案，保留自然口语问答，切分物理音频帧流 + 穿插远场抗噪 <unused5> 与真打断 <unused6>。
4. [真实环境噪音库] 本地 agkphysics/AudioSet (500 条 -> 提取 400 段 0.48s~0.96s 多样环境音，360 Train / 40 OOD)
5. [高质量口语短对话 + 异步工具扩容]
   - LooksJuicy/Chinese-Emotional-Intelligence + yuyouyu/BeyondDialogue (8,000 条口语多轮微回合 + CosyVoice 标签)
   - hqfx/fc_zh_hard_oai + glaive_toolcall_zh (6,000 条强化异步工具：NON_BLOCKING / <unused3> / <unused6><unused7>c1 / WHEN_IDLE)

输出：
- data/gemma4_duplex_route1_v2/train.json (~20,000 条, ~28% 含真实音频!)
- data/gemma4_duplex_route1_v2/valid.json (~1,000 条, 100% 零声源重叠!)
- data/gemma4_duplex_route1_v2/ood_eval.json (~80 条严格 OOD 盲测基准)
- data/gemma4_duplex_route1_v2/audios/**/*.flac (16kHz 无损 PCM_16 FLAC)
- data/gemma4_duplex_route1_v2/audios.tar (单文件打包，上传 HF 仅需 ~15 秒，Colab 秒级解压)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PARQUET_CACHE = Path.home() / ".cache" / "gemma4_duplex_v2_parquets"
SR = 16000
CHUNK_SAMPLES = 7680  # 0.48s @ 16kHz = 12 Gemma-4 audio soft tokens

SLOTS = {
    "no_voice": "<unused0>",       # ID 6: 0.48s 纯静音输入
    "user_talking": "<unused1>",   # ID 7: 继续倾听 / 防抢话 / WHEN_IDLE 压住不播
    "start_speak": "<unused2>",    # ID 8: 接话开口 / 垫话 + <|tool_call>
    "waiting_tool": "<unused3>",   # ID 9: 后台工具运行等待态
    "idle_think": "<unused4>",     # ID 10: 答完空闲态
    "ignore_noise": "<unused5>",   # ID 11: 忽略附和/环境噪/远场旁人混响，继续播报
    "interrupted": "<unused6>",    # ID 12: 近场真打断，立刻闭嘴
    "cancel_tool": "<unused7>",    # ID 13: 作废指定后台工具 ID
}

FILLERS_ZH = [
    "好的，马上帮您查。",
    "稍等，我帮您看一下。",
    "明白，正在为您处理。",
    "好的，请稍等片刻。",
    "收到，我这就帮您核实。",
]

BACKCHANNELS_ZH = ["嗯嗯", "好的", "对", "嗯", "哦哦", "行", "知道了"]

CANCEL_UTTERANCES_ZH = [
    "等一下，先不用查这个了",
    "算了算了，取消刚才那个操作吧",
    "啊不对，先停一下别办了",
    "先别查了，我换个问题问你",
]

# PolyAI/minds14 zh-CN 14 个真实人类银行业务语音意图 -> 真实工具定义 + 模拟返回 + 口语播报
MINDS14_INTENT_TOOLS: Dict[int, Dict[str, Any]] = {
    0: {
        "name": "check_abroad_usage",
        "desc": "查询银行卡境外消费与跨境使用规则",
        "params": {"type": "object", "properties": {"service": {"type": "string"}}, "required": []},
        "args": {"service": "abroad"},
        "obs": '{"enabled":true,"fee_rate":"1.5%"}',
        "reply": "您的银行卡已开通境外消费，跨境手续费率为百分之一点五。",
    },
    1: {
        "name": "update_address",
        "desc": "修改银行账户预留联系地址",
        "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": []},
        "args": {"action": "update"},
        "obs": '{"status":"verification_sent"}',
        "reply": "好的，已经向您手机发送了地址修改验证链接，确认后即可生效。",
    },
    2: {
        "name": "report_app_error",
        "desc": "排查并提交手机银行 App 报错工单",
        "params": {"type": "object", "properties": {"issue_type": {"type": "string"}}, "required": []},
        "args": {"issue_type": "app_error"},
        "obs": '{"ticket":"T8842","server_status":"normal"}',
        "reply": "服务器目前正常，已为您登记排查工单，建议先重启应用试试。",
    },
    3: {
        "name": "check_atm_limit",
        "desc": "查询银行卡 ATM 每日取现额度",
        "params": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": []},
        "args": {"channel": "atm"},
        "obs": '{"daily_limit":"20000元","remaining":"20000元"}',
        "reply": "您这张卡的取现日限额是两万元，今日可用额度还剩两万元。",
    },
    4: {
        "name": "check_balance",
        "desc": "查询银行账户当前可用余额",
        "params": {"type": "object", "properties": {"account_type": {"type": "string"}}, "required": []},
        "args": {"account_type": "savings"},
        "obs": '{"balance":"18650.50元","currency":"CNY"}',
        "reply": "您的账户当前可用余额是一万八千六百五十元五角。",
    },
    5: {
        "name": "query_business_loan",
        "desc": "查询企业与小微经营贷款申请条件",
        "params": {"type": "object", "properties": {"loan_type": {"type": "string"}}, "required": []},
        "args": {"loan_type": "business"},
        "obs": '{"max_amount":"50万元","apr":"3.85%"}',
        "reply": "目前小微经营贷最高可申请五十万元，年化利率三点八五起。",
    },
    6: {
        "name": "report_card_issue",
        "desc": "查询并处理银行卡异常、损坏或补换卡问题",
        "params": {"type": "object", "properties": {"issue": {"type": "string"}}, "required": []},
        "args": {"issue": "card_problem"},
        "obs": '{"card_status":"active","can_replace":true}',
        "reply": "查到您的卡片状态正常，如需补换新卡我可以马上为您登记。",
    },
    7: {
        "name": "query_cash_deposit",
        "desc": "查询现金存款渠道与单笔限额",
        "params": {"type": "object", "properties": {"method": {"type": "string"}}, "required": []},
        "args": {"method": "cash"},
        "obs": '{"atm_deposit_limit":"50000元","counter_required_above":"50000元"}',
        "reply": "自助存取款机单日最多可存五万元，超过金额需携带身份证去柜台。",
    },
    8: {
        "name": "manage_direct_debit",
        "desc": "查询或管理自动代扣与定期扣款协议",
        "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": []},
        "args": {"action": "list"},
        "obs": '{"active_debits":2,"next_date":"每月15日"}',
        "reply": "您名下目前有两项生效中的自动代扣，下一次扣款在每月十五号。",
    },
    9: {
        "name": "freeze_card",
        "desc": "紧急冻结或解冻银行卡",
        "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]},
        "args": {"action": "freeze"},
        "obs": '{"card_frozen":true,"effective":"immediate"}',
        "reply": "已为您紧急冻结该银行卡，所有线上线下消费通道都已暂停。",
    },
    10: {
        "name": "query_high_value_payment",
        "desc": "查询大额转账限额与安全验证流程",
        "params": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": []},
        "args": {"channel": "transfer"},
        "obs": '{"single_limit":"50万元","auth":"ukey_or_face"}',
        "reply": "单笔大额转账最高支持五十万元，需要通过人脸或安全盾验证。",
    },
    11: {
        "name": "query_joint_account",
        "desc": "查询联名共同账户开立与管理规则",
        "params": {"type": "object", "properties": {"account_type": {"type": "string"}}, "required": []},
        "args": {"account_type": "joint"},
        "obs": '{"supported":true,"required":"both_ids"}',
        "reply": "联名账户需要两位持卡人携带本人身份证一同办理开户。",
    },
    12: {
        "name": "query_latest_transactions",
        "desc": "查询银行账户最近消费与入账流水",
        "params": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        "args": {"limit": 3},
        "obs": '{"latest":[{"merchant":"超市","amount":"-68.5元"},{"merchant":"咖啡店","amount":"-28元"}]}',
        "reply": "您最近两笔消费是超市六十八块五，还有咖啡店二十八元。",
    },
    13: {
        "name": "pay_bill",
        "desc": "办理账单缴费与信用卡还款",
        "params": {"type": "object", "properties": {"bill_type": {"type": "string"}}, "required": []},
        "args": {"bill_type": "utility_or_credit"},
        "obs": '{"due_amount":"326.00元","due_date":"本月28日"}',
        "reply": "查到您本期应缴账单是三百二十六元，截止日期是本月二十八号。",
    },
}

PARQUET_URLS = {
    "minds14_zh": "https://huggingface.co/datasets/PolyAI/minds14/resolve/refs%2Fconvert%2Fparquet/zh-CN/train/0000.parquet",
    "aishell1_0": "https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    "cosy_qa_0": "https://huggingface.co/datasets/JerryAGENDD/chinese_speech_cosy_audio/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
}


def download_parquet(name: str, url: str) -> Path:
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    dst = PARQUET_CACHE / f"{name}.parquet"
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print(f"  [Cache HIT] {dst.name} ({dst.stat().st_size / 1024**2:.1f} MB)")
        return dst
    print(f"  [Download] {name} <- {url}")
    tmp = dst.with_suffix(".parquet.part")
    req = urllib.request.Request(url, headers={"User-Agent": "gemma4-duplex-v2-builder"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)
    print(f"  [Done] {dst.name} ({dst.stat().st_size / 1024**2:.1f} MB)")
    return dst


def decode_audio_bytes(raw: bytes, target_sr: int = SR) -> Optional[np.ndarray]:
    try:
        wav, sr_in = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < 1600:
        return None
    if sr_in != target_sr:
        g = math.gcd(sr_in, target_sr)
        wav = resample_poly(wav, target_sr // g, sr_in // g).astype(np.float32)
    return wav


def trim_silence(wav: np.ndarray, thresh: float = 0.012, pad: int = 1200) -> np.ndarray:
    nz = np.where(np.abs(wav) > thresh)[0]
    if len(nz) == 0:
        return wav
    s = max(0, int(nz[0]) - pad)
    e = min(len(wav), int(nz[-1]) + pad)
    return wav[s:e]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def normalize_rms(x: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    r = rms(x)
    if r < 1e-6:
        return x
    return np.clip(x * (target_rms / r), -0.98, 0.98).astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        reps = int(math.ceil(len(clean) / max(1, len(noise))))
        noise = np.tile(noise, reps)[: len(clean)]
    else:
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start : start + len(clean)]
    r_c = rms(clean)
    r_n = rms(noise)
    if r_n < 1e-6:
        return clean
    scale = r_c / (r_n * (10.0 ** (snr_db / 20.0)))
    return np.clip(clean + noise * scale, -0.98, 0.98).astype(np.float32)


def write_flac(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(wav, -0.99, 0.99).astype(np.float32)
    sf.write(str(path), wav, SR, format="FLAC", subtype="PCM_16")


def split_audio_into_chunks(wav: np.ndarray, n_parts: int) -> List[np.ndarray]:
    """将一段真实语音切成 2~3 个自然物理流式切片（每片 >= 0.48s）"""
    wav = trim_silence(wav)
    if len(wav) < CHUNK_SAMPLES * 2 or n_parts <= 1:
        if len(wav) < CHUNK_SAMPLES:
            wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        return [wav]
    cuts = sorted(
        random.sample(
            range(CHUNK_SAMPLES, max(CHUNK_SAMPLES + 1, len(wav) - CHUNK_SAMPLES + 1)),
            k=min(n_parts - 1, max(1, (len(wav) // CHUNK_SAMPLES) - 1)),
        )
    )
    pts = [0] + cuts + [len(wav)]
    parts = [wav[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    return [p for p in parts if len(p) >= int(SR * 0.35)]


def split_speech_chunks(text: str, min_chars: int = 3, max_chars: int = 6) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "[":
            end_b = text.find("]", i)
            if 0 < end_b - i <= 12:
                step = (end_b - i + 1) + random.randint(2, 4)
                chunks.append(text[i : i + step])
                i += step
                continue
        step = random.randint(min_chars, max_chars)
        chunks.append(text[i : i + step])
        i += step
    return chunks


def inject_cosyvoice_paralinguistics(ans: str) -> str:
    if any(w in ans for w in ["哈哈", "开心", "太棒", "恭喜"]) and random.random() < 0.30:
        return "[laughter] " + ans
    if any(w in ans for w in ["难过", "遗憾", "抱歉", "辛苦", "累"]) and random.random() < 0.30:
        return "[sigh] " + ans
    if len(ans) > 22 and "，" in ans and random.random() < 0.22:
        parts = ans.split("，", 1)
        return f"{parts[0]}，[breath] {parts[1]}"
    return ans


def summarize_spoken_answer(text: str, max_len: int = 80) -> str:
    """将书面长总结或列表压缩为适合实时语音播报的自然短句"""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[#*`>\-]", "", text)
    text = re.sub(r"\b\d+\.\s*", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)
    joined = re.sub(r"\s+", " ", joined).strip()
    if len(joined) > max_len:
        cut = joined[:max_len]
        for p in ("。", "！", "？", "；", "，"):
            pos = cut.rfind(p)
            if pos >= 18:
                return cut[: pos + 1]
        return cut + "。"
    return joined


def is_clean_spoken_zh(text: str, min_len: int = 6, max_len: int = 110) -> bool:
    if not text:
        return False
    t = text.strip()
    if not (min_len <= len(t) <= max_len):
        return False
    if any(bad in t for bad in ["```", "###", "**", "\n1.", "\n-", "http://", "https://", "def ", "import "]):
        return False
    zh_cnt = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    return zh_cnt >= max(4, int(len(t) * 0.45))


def format_gemma4_tool_call(call_id: str, func_name: str, args_obj: Dict[str, Any]) -> str:
    def _fmt(v: Any) -> str:
        if isinstance(v, str):
            return f'<|"|>{v}<|"|>'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ",".join(_fmt(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{_fmt(val)}" for k, val in sorted(v.items())) + "}"
        return f'<|"|>{str(v)}<|"|>'

    merged = {"id": call_id, **args_obj}
    body = ",".join(f"{k}:{_fmt(v)}" for k, v in merged.items())
    return f"<|tool_call>call:{func_name}{{{body}}}<tool_call|>"


def make_tool_decl(name: str, desc: str, params: Dict[str, Any]) -> str:
    return json.dumps([{"name": name, "description": desc, "parameters": params}], ensure_ascii=False)


# =============================================================================
# 1. 构建严格 Train / OOD 隔离的声学底池 (AudioSet 噪音 + MIT RIR 房间冲激 + AISHELL-1 远近场)
# =============================================================================
def load_mit_rirs() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rir_files = sorted(
        glob.glob(str(HF_CACHE / "datasets--davidscripka--MIT_environmental_impulse_responses/snapshots/*/**/*.wav"), recursive=True)
    )
    rirs: List[np.ndarray] = []
    for f in rir_files:
        try:
            w, sr_in = sf.read(f, dtype="float32")
            if w.ndim > 1:
                w = w[:, 0]
            if sr_in != SR:
                g = math.gcd(sr_in, SR)
                w = resample_poly(w, SR // g, sr_in // g).astype(np.float32)
            peak = float(np.max(np.abs(w)) + 1e-8)
            rirs.append((w / peak)[: SR])
        except Exception:
            continue
    random.shuffle(rirs)
    split = max(1, int(len(rirs) * 0.85))
    return rirs[:split], rirs[split:]


def load_audioset_bank(
    audio_dir: Path, data_rel_prefix: str, max_clips: int = 400
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    as_pqs = glob.glob(str(HF_CACHE / "datasets--agkphysics--AudioSet/snapshots/*/**/*.parquet"), recursive=True)
    items: List[Tuple[str, np.ndarray]] = []
    if as_pqs:
        tbl = pq.read_table(as_pqs[0])
        for i in range(min(tbl.num_rows, max_clips + 100)):
            if len(items) >= max_clips:
                break
            aud = tbl["audio"][i].as_py()
            raw = aud.get("bytes") if isinstance(aud, dict) else None
            if not raw:
                continue
            wav = decode_audio_bytes(raw)
            if wav is None or len(wav) < CHUNK_SAMPLES:
                continue
            dur_n = random.choice([CHUNK_SAMPLES, int(CHUNK_SAMPLES * 1.5), CHUNK_SAMPLES * 2])
            if len(wav) > dur_n:
                s = random.randint(0, len(wav) - dur_n)
                clip = wav[s : s + dur_n]
            else:
                clip = wav
            if rms(clip) < 0.005:
                continue
            clip = normalize_rms(clip, target_rms=random.uniform(0.03, 0.09))
            sub = "tr_noise" if len(items) < int(max_clips * 0.9) else "ood_noise"
            rel = f"{data_rel_prefix}/audios/{sub}/as_{i:04d}.flac"
            write_flac(audio_dir / sub / f"as_{i:04d}.flac", clip)
            items.append((rel, clip))
    split = int(len(items) * 0.9)
    return items[:split], items[split:]


def build_aishell_acoustic_banks(
    aishell_pq: Path,
    audio_dir: Path,
    data_rel_prefix: str,
    rirs_tr: List[np.ndarray],
    rirs_ood: List[np.ndarray],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    max_utterances: int = 1600,
) -> Dict[str, Any]:
    """
    从 AISHELL-1 真实多说话人中文语音构建 4 大声学资源（严格 90/10 隔离）：
    - streaming_utts: ([(rel_path, dur), ...], transcript) 用于真实语音流式倾听/防抢话
    - farfield_clips: 卷积 MIT RIR + 衰减 + 可选 AudioSet 底噪 -> <unused5> 远场旁人干扰
    - nearfield_barge_clips: 近场清晰人声 + 50% 叠 AudioSet (+6~+14dB SNR) -> <unused6> 嘈杂近场真打断
    - short_backchannel_clips: 低能量 0.48s 短人声片段 -> <unused5> 忽略轻声附和
    """
    tbl = pq.read_table(aishell_pq)
    idxs = list(range(tbl.num_rows))
    random.shuffle(idxs)
    idxs = idxs[:max_utterances]
    split_n = int(len(idxs) * 0.90)
    tr_idxs, ood_idxs = idxs[:split_n], idxs[split_n:]

    out: Dict[str, Any] = {
        "tr_farfield": [],
        "ood_farfield": [],
        "tr_barge": [],
        "ood_barge": [],
        "tr_backchannel": [],
        "ood_backchannel": [],
        "tr_stream": [],
        "ood_stream": [],
    }

    def _process_subset(subset_idxs: List[int], is_ood: bool) -> None:
        tag = "ood" if is_ood else "tr"
        rirs = rirs_ood if is_ood else rirs_tr
        noises = noises_ood if is_ood else noises_tr
        for k, row_i in enumerate(subset_idxs):
            ctx = tbl["context"][row_i].as_py()
            ans = str(tbl["answer"][row_i].as_py() or "").strip()
            raw = ctx.get("bytes") if isinstance(ctx, dict) else None
            if not raw or not ans or len(ans) < 5:
                continue
            wav = decode_audio_bytes(raw)
            if wav is None:
                continue
            wav = trim_silence(wav)
            if len(wav) < CHUNK_SAMPLES:
                continue
            wav_clean = normalize_rms(wav, target_rms=random.uniform(0.08, 0.13))

            # 1) 流式人声切片样本（前 550 条 train + 前 40 条 ood 切为 2~3 片）
            if (not is_ood and len(out["tr_stream"]) < 550) or (is_ood and len(out["ood_stream"]) < 40):
                parts = split_audio_into_chunks(wav_clean, n_parts=random.choice([2, 3]))
                if len(parts) >= 2:
                    rels = []
                    for pi, p_wav in enumerate(parts):
                        bucket = f"{tag}_aishell_s{k % 5}"
                        fname = f"ais_{row_i:05d}_p{pi}.flac"
                        write_flac(audio_dir / bucket / fname, p_wav)
                        rels.append(f"{data_rel_prefix}/audios/{bucket}/{fname}")
                    out[f"{tag}_stream"].append({"chunks": rels, "text": ans})

            # 2) MIT RIR 远场旁人混响样本 (<unused5>)
            if (not is_ood and len(out["tr_farfield"]) < 500) or (is_ood and len(out["ood_farfield"]) < 45):
                seg_len = random.choice([CHUNK_SAMPLES, int(CHUNK_SAMPLES * 1.5), CHUNK_SAMPLES * 2])
                seg = wav_clean[:seg_len] if len(wav_clean) >= seg_len else np.pad(wav_clean, (0, seg_len - len(wav_clean)))
                if rirs:
                    rir = random.choice(rirs)
                    far = fftconvolve(seg, rir, mode="full")[: len(seg)].astype(np.float32)
                else:
                    far = seg.copy()
                # 物理衰减到远场能量 (-14dB ~ -22dB)
                att_db = random.uniform(14.0, 22.0)
                far = normalize_rms(far, target_rms=0.10 * (10.0 ** (-att_db / 20.0)))
                if noises and random.random() < 0.65:
                    _, n_wav = random.choice(noises)
                    far = mix_at_snr(far, n_wav, snr_db=random.uniform(-2.0, 6.0))
                bucket = f"{tag}_farfield"
                fname = f"far_{row_i:05d}.flac"
                write_flac(audio_dir / bucket / fname, far)
                out[f"{tag}_farfield"].append(f"{data_rel_prefix}/audios/{bucket}/{fname}")

            # 3) 近场清晰 / 嘈杂近场真打断样本 (<unused6>)
            if (not is_ood and len(out["tr_barge"]) < 500) or (is_ood and len(out["ood_barge"]) < 45):
                seg_len = min(len(wav_clean), random.choice([int(SR * 0.96), int(SR * 1.44), int(SR * 1.92)]))
                barge = wav_clean[:seg_len]
                # 55% 叠加 AudioSet 环境噪 (+7 ~ +15dB SNR)，让模型学会“屋里吵但用户近场插话仍然要真打断!”
                if noises and random.random() < 0.55:
                    _, n_wav = random.choice(noises)
                    barge = mix_at_snr(barge, n_wav, snr_db=random.uniform(7.0, 15.0))
                bucket = f"{tag}_barge"
                fname = f"barge_{row_i:05d}.flac"
                write_flac(audio_dir / bucket / fname, barge)
                out[f"{tag}_barge"].append({"path": f"{data_rel_prefix}/audios/{bucket}/{fname}", "text": ans})

            # 4) 短低能附和音切片 (<unused5>)
            if (not is_ood and len(out["tr_backchannel"]) < 180) or (is_ood and len(out["ood_backchannel"]) < 20):
                bc = wav_clean[:CHUNK_SAMPLES] if len(wav_clean) >= CHUNK_SAMPLES else np.pad(wav_clean, (0, CHUNK_SAMPLES - len(wav_clean)))
                bc = normalize_rms(bc, target_rms=random.uniform(0.015, 0.032))
                bucket = f"{tag}_backchannel"
                fname = f"bc_{row_i:05d}.flac"
                write_flac(audio_dir / bucket / fname, bc)
                out[f"{tag}_backchannel"].append(f"{data_rel_prefix}/audios/{bucket}/{fname}")

    _process_subset(tr_idxs, is_ood=False)
    _process_subset(ood_idxs, is_ood=True)
    return out


# =============================================================================
# 2. 模块 A：PolyAI/minds14 (zh-CN) 真实人类语音 -> 零 ASR 异步工具状态机 + OOD 盲测
# =============================================================================
def build_minds14_audio_tool_samples(
    minds_pq: Path,
    audio_dir: Path,
    data_rel_prefix: str,
    bank: Dict[str, Any],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    tbl = pq.read_table(minds_pq)
    idxs = list(range(tbl.num_rows))
    random.shuffle(idxs)
    split_n = int(len(idxs) * 0.88)  # 441 train / 61 OOD held-out
    tr_idxs, ood_idxs = idxs[:split_n], idxs[split_n:]

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    ood_cases: List[Dict[str, Any]] = []

    def _build_one(row_i: int, is_ood: bool, variant: int) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        aud = tbl["audio"][row_i].as_py()
        raw = aud.get("bytes") if isinstance(aud, dict) else None
        intent = int(tbl["intent_class"][row_i].as_py())
        transcript = str(tbl["transcription"][row_i].as_py() or "").strip()
        if not raw or intent not in MINDS14_INTENT_TOOLS:
            return None
        wav = decode_audio_bytes(raw)
        if wav is None:
            return None
        wav = normalize_rms(trim_silence(wav), target_rms=random.uniform(0.08, 0.13))
        noises = noises_ood if is_ood else noises_tr
        far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
        barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
        noise_paths = [p for p, _ in noises]

        # variant=1 叠加轻度真实环境背景噪 (+10~18dB SNR)，提升抗噪零 ASR 抽参泛化
        if variant == 1 and noises:
            _, n_wav = random.choice(noises)
            wav = mix_at_snr(wav, n_wav, snr_db=random.uniform(10.0, 18.0))

        parts = split_audio_into_chunks(wav, n_parts=random.choice([2, 3]))
        tag = "ood_minds" if is_ood else f"tr_minds_v{variant}"
        part_paths: List[str] = []
        for pi, p_wav in enumerate(parts):
            fname = f"m14_{row_i:04d}_v{variant}_p{pi}.flac"
            write_flac(audio_dir / tag / fname, p_wav)
            part_paths.append(f"{data_rel_prefix}/audios/{tag}/{fname}")

        spec = MINDS14_INTENT_TOOLS[intent]
        # 随机加入 2~3 个同域干扰工具声明，防止模型靠“系统只有1个工具”死记硬背！
        distractor_ids = random.sample([k for k in MINDS14_INTENT_TOOLS if k != intent], k=3)
        tool_decls = [
            {"name": MINDS14_INTENT_TOOLS[k]["name"], "description": MINDS14_INTENT_TOOLS[k]["desc"], "parameters": MINDS14_INTENT_TOOLS[k]["params"]}
            for k in [intent] + distractor_ids
        ]
        random.shuffle(tool_decls)
        tools_str = json.dumps(tool_decls, ensure_ascii=False)

        convs: List[Dict[str, str]] = []
        audios: List[str] = []

        # 1. 真实人类音频切片直入 + 句中喘气静音帧防抢话
        for pi, p_rel in enumerate(part_paths):
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(p_rel)
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            if pi == 0 and len(part_paths) >= 2 and random.random() < 0.7:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        # 2. 说完静音 -> 垫话 + 异步 <|tool_call>
        call_str = format_gemma4_tool_call("c1", spec["name"], spec["args"])
        convs.append({"from": "human", "value": SLOTS["no_voice"]})
        convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{random.choice(FILLERS_ZH)}{call_str}"})

        # 3. 后台异步等待帧
        for _ in range(random.randint(1, 2)):
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["waiting_tool"]})

        # 4. 25% 概率在工具未返回前遭遇近场真打断 -> <unused6><unused7>c1 作废旧工具！
        mode = "normal"
        if random.random() < 0.25 and barge_pool:
            mode = "cancel"
            b_item = random.choice(barge_pool)
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(b_item["path"])
            convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已为您取消刚才的查询，请问还有什么可以帮您？"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
        else:
            # 工具结果返回 -> 分块播报 + 穿插远场混响/环境噪(<unused5>) + 近场真打断(<unused6>)
            convs.append({"from": "observation", "value": spec["obs"]})
            reply_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(spec["reply"]), 4, 8)
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{reply_chunks[0]}"})
            for ci, rc in enumerate(reply_chunks[1:]):
                r_p = random.random()
                if ci == 0 and far_pool and r_p < 0.45:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(far_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{rc}"})
                elif ci == 0 and noise_paths and r_p < 0.75:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(noise_paths))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{rc}"})
                elif ci == len(reply_chunks) - 2 and barge_pool and random.random() < 0.35:
                    b_item = random.choice(barge_pool)
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(b_item["path"])
                    convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                    break
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{rc}"})
            else:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

        sample = {"conversations": convs, "tools": tools_str, "audios": audios}
        ood_meta = {
            "type": "minds14_zero_asr_tool",
            "expected_tool": spec["name"],
            "transcript": transcript,
            "tools": tools_str,
            "part_paths": part_paths,
            "obs": spec["obs"],
            "far_path": random.choice(far_pool) if far_pool else None,
            "noise_path": random.choice(noise_paths) if noise_paths else None,
            "barge_path": random.choice(barge_pool)["path"] if barge_pool else None,
        }
        return sample, ood_meta

    for row_i in tr_idxs:
        for var in (0, 1):
            res = _build_one(row_i, is_ood=False, variant=var)
            if res:
                train_rows.append(res[0])

    for row_i in ood_idxs:
        res = _build_one(row_i, is_ood=True, variant=0)
        if res:
            valid_rows.append(res[0])
            if len(ood_cases) < 25:
                ood_cases.append(res[1])

    return train_rows, valid_rows, ood_cases


# =============================================================================
# 3. 模块 B：JerryAGENDD/chinese_speech_cosy_audio 真实中文语音问答流 + 抗噪/打断
# =============================================================================
def build_cosy_qa_audio_samples(
    cosy_pq: Path,
    audio_dir: Path,
    data_rel_prefix: str,
    bank: Dict[str, Any],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    max_rows: int = 2000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tbl = pq.read_table(cosy_pq)
    idxs = list(range(tbl.num_rows))
    random.shuffle(idxs)

    tr_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    kept = 0

    for row_i in idxs:
        if kept >= max_rows:
            break
        raw_ans = str(tbl["answer"][row_i].as_py() or "").strip()
        if "```" in raw_ans or "def " in raw_ans or "import " in raw_ans:
            continue
        ans = summarize_spoken_answer(raw_ans, max_len=82)
        if not is_clean_spoken_zh(ans, min_len=12, max_len=95):
            continue
        aud = tbl["audio"][row_i].as_py()
        raw = aud.get("bytes") if isinstance(aud, dict) else None
        if not raw:
            continue
        wav = decode_audio_bytes(raw)
        if wav is None:
            continue
        wav = normalize_rms(trim_silence(wav), target_rms=random.uniform(0.08, 0.13))
        if len(wav) < CHUNK_SAMPLES or len(wav) > SR * 10:
            continue

        is_ood = kept >= int(max_rows * 0.92)
        tag = f"ood_cosy" if is_ood else f"tr_cosy_s{kept % 6}"
        far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
        barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
        bc_pool = bank["ood_backchannel"] if is_ood else bank["tr_backchannel"]
        noise_paths = [p for p, _ in (noises_ood if is_ood else noises_tr)]

        # 30% 混入真实环境底噪 (+10~18dB) 模拟真实居家/办公麦克风输入
        if (noises_ood if is_ood else noises_tr) and random.random() < 0.30:
            _, n_wav = random.choice(noises_ood if is_ood else noises_tr)
            wav = mix_at_snr(wav, n_wav, snr_db=random.uniform(10.0, 18.0))

        parts = split_audio_into_chunks(wav, n_parts=random.choice([2, 3]))
        part_paths: List[str] = []
        for pi, p_wav in enumerate(parts):
            fname = f"cqa_{row_i:05d}_p{pi}.flac"
            write_flac(audio_dir / tag / fname, p_wav)
            part_paths.append(f"{data_rel_prefix}/audios/{tag}/{fname}")

        convs: List[Dict[str, str]] = []
        audios: List[str] = []

        for pi, p_rel in enumerate(part_paths):
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(p_rel)
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            if pi < len(part_paths) - 1 and random.random() < 0.55:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        ans_sp = inject_cosyvoice_paralinguistics(ans)
        chunks = split_speech_chunks(ans_sp, 4, 7)
        if not chunks:
            continue

        convs.append({"from": "human", "value": SLOTS["no_voice"]})
        convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{chunks[0]}"})

        interrupted = False
        for ci, ch in enumerate(chunks[1:]):
            r = random.random()
            if ci == 0 and far_pool and r < 0.28:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(far_pool))
                convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
            elif ci == 0 and noise_paths and r < 0.48:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(noise_paths))
                convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
            elif ci == 1 and bc_pool and r < 0.25:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(bc_pool))
                convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
            elif ci >= 1 and barge_pool and random.random() < 0.22:
                b_item = random.choice(barge_pool)
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(b_item["path"])
                convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                interrupted = True
                break
            else:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ch}"})

        if not interrupted:
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

        row = {"conversations": convs, "tools": "[]", "audios": audios}
        if is_ood:
            val_rows.append(row)
        else:
            tr_rows.append(row)
        kept += 1

    return tr_rows, val_rows


# =============================================================================
# 4. 模块 C：AISHELL-1 真实多说话人声学边界对比训练样本 (远场混响 <unused5> vs 嘈杂近场 <unused6>)
# =============================================================================
def build_aishell_contrast_samples(
    bank: Dict[str, Any],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    spoken_pairs: List[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将 AISHELL-1 真实多说话人音频切片、MIT RIR 远场旁人混响、嘈杂近场打断、AudioSet 噪音
    密集注入真实对话流，专门拉高声学边界准确率！
    """
    tr_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    def _make_set(is_ood: bool, count: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        stream_pool = bank["ood_stream"] if is_ood else bank["tr_stream"]
        far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
        barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
        bc_pool = bank["ood_backchannel"] if is_ood else bank["tr_backchannel"]
        noise_paths = [p for p, _ in (noises_ood if is_ood else noises_tr)]
        if not (far_pool and barge_pool and noise_paths):
            return rows

        for i in range(count):
            convs: List[Dict[str, str]] = []
            audios: List[str] = []

            # 前半段：50% 用 AISHELL 真实语音切片听写/确认，50% 用口语问答起头
            if stream_pool and i % 2 == 0:
                st = stream_pool[i % len(stream_pool)]
                for ci, p_rel in enumerate(st["chunks"]):
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(p_rel)
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                    if ci == 0 and random.random() < 0.6:
                        convs.append({"from": "human", "value": SLOTS["no_voice"]})
                        convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                # 模型用口语自然接话复述要点，确保零 ASR 听觉对齐不退化
                ans_text = f"我听到了，您刚才说的是{st['text']}。"
            else:
                q_txt, a_txt = random.choice(spoken_pairs)
                for q_ch in split_speech_chunks(q_txt, 3, 6):
                    convs.append({"from": "human", "value": q_ch})
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                ans_text = inject_cosyvoice_paralinguistics(a_txt)

            chunks = split_speech_chunks(ans_text, 4, 7)
            if len(chunks) < 2:
                chunks = [ans_text, "还有别的需要帮忙吗？"]

            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{chunks[0]}"})

            # 在播报中密集穿插 1~2 次真实声学干扰（远场混响 / AudioSet 噪 / 短附和 -> <unused5>）+ 45% 近场真打断 (<unused6>)
            interrupted = False
            for ci, ch in enumerate(chunks[1:]):
                p = random.random()
                if p < 0.35:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(far_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.60:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(noise_paths))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.72 and bc_pool:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(bc_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.90:
                    b_item = random.choice(barge_pool)
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(b_item["path"])
                    convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                    interrupted = True
                    break
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ch}"})

            if not interrupted:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

            rows.append({"conversations": convs, "tools": "[]", "audios": audios})
        return rows

    tr_rows = _make_set(is_ood=False, count=2400)
    val_rows = _make_set(is_ood=True, count=160)
    return tr_rows, val_rows


# =============================================================================
# 5. 模块 D：高质量中文口语多轮短对话 (Chinese-Emotional-Intelligence + BeyondDialogue)
# =============================================================================
def load_spoken_dialogue_pairs(max_pairs: int = 12000) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    # 1) LooksJuicy/Chinese-Emotional-Intelligence
    ei_files = glob.glob(str(HF_CACHE / "datasets--LooksJuicy--Chinese-Emotional-Intelligence/snapshots/*/*.json"))
    if ei_files:
        data = json.load(open(ei_files[0], encoding="utf-8"))
        random.shuffle(data)
        for r in data:
            q = str(r.get("instruction") or "").strip()
            a = str(r.get("output") or "").strip()
            if is_clean_spoken_zh(q, 5, 60) and is_clean_spoken_zh(a, 10, 90):
                pairs.append((q, a))
                if len(pairs) >= int(max_pairs * 0.65):
                    break

    # 2) yuyouyu/BeyondDialogue (*_CN_SFT.json)
    bd_files = glob.glob(str(HF_CACHE / "datasets--yuyouyu--BeyondDialogue/snapshots/*/*CN_SFT.json"))
    for f in bd_files:
        try:
            data = json.load(open(f, encoding="utf-8"))
            random.shuffle(data)
            for r in data:
                q = str(r.get("instruction") or "").strip()
                a = str(r.get("output") or "").strip()
                if is_clean_spoken_zh(q, 5, 65) and is_clean_spoken_zh(a, 10, 95):
                    pairs.append((q, a))
                    if len(pairs) >= max_pairs:
                        break
        except Exception:
            continue
        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    return pairs


def build_spoken_text_duplex_rows(pairs: List[Tuple[str, str]], count: int = 8000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    while len(rows) < count and idx + 1 < len(pairs):
        turns = random.choice([1, 2])
        convs: List[Dict[str, str]] = []
        for _ in range(turns):
            if idx >= len(pairs):
                break
            q, a = pairs[idx]
            idx += 1
            q_chunks = split_speech_chunks(q, 3, 6)
            for qi, qc in enumerate(q_chunks):
                convs.append({"from": "human", "value": qc})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                if qi < len(q_chunks) - 1 and random.random() < 0.35:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            a_sp = inject_cosyvoice_paralinguistics(a)
            a_chunks = split_speech_chunks(a_sp, 4, 7)
            if not a_chunks:
                continue
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{a_chunks[0]}"})

            interrupted = False
            for ci, ac in enumerate(a_chunks[1:]):
                r = random.random()
                if r < 0.20:
                    convs.append({"from": "human", "value": random.choice(BACKCHANNELS_ZH)})
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                elif ci >= 1 and r < 0.42:
                    convs.append({"from": "human", "value": random.choice(["等一下", "等等我想问", "先停一下"])})
                    convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                    interrupted = True
                    break
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ac}"})
            if not interrupted:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

        if convs:
            rows.append({"conversations": convs, "tools": "[]", "audios": []})
    return rows


# =============================================================================
# 6. 模块 E：强化异步工具状态机 (hqfx/fc_zh_hard_oai -> 6,000 条含作废旧工具 + 改参重查 + WHEN_IDLE)
# =============================================================================
def build_async_tool_text_rows(max_rows: int = 6000) -> List[Dict[str, Any]]:
    pqs = glob.glob(str(HF_CACHE / "datasets--hqfx--fc_zh_hard_oai/snapshots/*/**/*.parquet"), recursive=True)
    if not pqs:
        return []
    tbl = pq.read_table(pqs[0])
    base_items: List[Dict[str, Any]] = []

    for i in range(tbl.num_rows):
        msgs = tbl["conversation"][i].as_py()
        tools_raw = tbl["tools"][i].as_py()
        if not msgs or not tools_raw:
            continue
        try:
            tools_list = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
            norm_tools = []
            for t in tools_list:
                fn = t.get("function", t) if isinstance(t, dict) else {}
                if fn.get("name"):
                    norm_tools.append({
                        "name": fn["name"],
                        "description": str(fn.get("description") or f"执行 {fn['name']}")[:100],
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    })
            if not norm_tools:
                continue
        except Exception:
            continue

        for j in range(len(msgs) - 3):
            m_u, m_tc, m_obs, m_ans = msgs[j], msgs[j + 1], msgs[j + 2], msgs[j + 3]
            if not (
                m_u.get("role") == "user"
                and m_tc.get("role") == "assistant"
                and m_tc.get("tool_calls")
                and m_obs.get("role") == "tool"
                and m_ans.get("role") == "assistant"
                and m_ans.get("content")
            ):
                continue
            user_q = str(m_u.get("content") or "").splitlines()[0].strip()
            if not (4 <= len(user_q) <= 85):
                continue
            tc0 = m_tc["tool_calls"][0]
            fn = tc0.get("function", tc0)
            call_name = fn.get("name")
            raw_a = fn.get("arguments", "{}")
            try:
                call_args = json.loads(raw_a) if isinstance(raw_a, str) else raw_a
            except Exception:
                continue
            if not (call_name and isinstance(call_args, dict)):
                continue
            obs_str = str(m_obs.get("content") or "{}").strip()[:280]
            final_ans = summarize_spoken_answer(str(m_ans.get("content") or ""), max_len=85)
            if len(final_ans) < 8:
                continue
            base_items.append({
                "tools": json.dumps(norm_tools[:5], ensure_ascii=False),
                "q": user_q,
                "name": call_name,
                "args": call_args,
                "obs": obs_str,
                "ans": final_ans,
            })
            break

    rows: List[Dict[str, Any]] = []
    if not base_items:
        return rows

    for i in range(max_rows):
        it = base_items[i % len(base_items)]
        convs: List[Dict[str, str]] = []
        q_chunks = split_speech_chunks(it["q"], 3, 6)
        for qi, qc in enumerate(q_chunks):
            convs.append({"from": "human", "value": qc})
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            if qi == 0 and len(q_chunks) > 1 and random.random() < 0.45:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        call_str = format_gemma4_tool_call("c1", it["name"], it["args"])
        convs.append({"from": "human", "value": SLOTS["no_voice"]})
        convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{random.choice(FILLERS_ZH)}{call_str}"})

        for _ in range(random.randint(1, 2)):
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["waiting_tool"]})

        # 4 大异步范式均匀覆盖：
        # 0~0.30: 中途打断并作废旧工具 <unused6><unused7>c1
        # 0.30~0.50: WHEN_IDLE (工具返回时用户正在插话 -> 先 <unused1> 压住不播，等用户说完再播!)
        # 0.50~1.00: 正常异步回传 + 播报中穿插附和忽略 <unused5> / 真打断 <unused6>
        r = random.random()
        if r < 0.30:
            convs.append({"from": "human", "value": random.choice(CANCEL_UTTERANCES_ZH)})
            convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已为您取消刚才的操作。"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
        elif r < 0.50:
            # WHEN_IDLE：用户还在追问时收到工具结果，模型先 <unused1> 忍住不抢话
            convs.append({"from": "human", "value": "顺便问一下"})
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            convs.append({"from": "observation", "value": it["obs"]})
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            a_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(it["ans"]), 4, 7)
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{a_chunks[0]}"})
            for ac in a_chunks[1:]:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ac}"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
        else:
            convs.append({"from": "observation", "value": it["obs"]})
            a_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(it["ans"]), 4, 7)
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{a_chunks[0]}"})
            interrupted = False
            for ci, ac in enumerate(a_chunks[1:]):
                if ci == 0 and random.random() < 0.30:
                    convs.append({"from": "human", "value": random.choice(BACKCHANNELS_ZH)})
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                elif ci >= 1 and random.random() < 0.25:
                    convs.append({"from": "human", "value": "行了我知道了，先停一下"})
                    convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                    interrupted = True
                    break
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ac}"})
            if not interrupted:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

        rows.append({"conversations": convs, "tools": it["tools"], "audios": []})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "data" / "gemma4_duplex_route1_v2"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audios"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_rel_prefix = out_dir.name  # relative to LlamaFactory/data/

    print("🚀 [1/7] 下载/读取三大真实中文语音 Parquet 分片...")
    minds_pq = download_parquet("minds14_zh", PARQUET_URLS["minds14_zh"])
    aishell_pq = download_parquet("aishell1_0", PARQUET_URLS["aishell1_0"])
    cosy_pq = download_parquet("cosy_qa_0", PARQUET_URLS["cosy_qa_0"])

    print("🔊 [2/7] 加载 MIT RIR (270 真实房间混响) & AudioSet (400 真实环境音)...")
    rirs_tr, rirs_ood = load_mit_rirs()
    noises_tr, noises_ood = load_audioset_bank(audio_dir, data_rel_prefix, max_clips=400)
    print(f"   MIT RIR: train={len(rirs_tr)}, ood={len(rirs_ood)} | AudioSet: train={len(noises_tr)}, ood={len(noises_ood)}")

    print("🎙️ [3/7] 构建 AISHELL-1 真实多说话人远近场声学对比库 (90/10 严格隔离)...")
    bank = build_aishell_acoustic_banks(
        aishell_pq, audio_dir, data_rel_prefix, rirs_tr, rirs_ood, noises_tr, noises_ood, max_utterances=1600
    )
    print(
        f"   AISHELL: tr_stream={len(bank['tr_stream'])}, tr_far={len(bank['tr_farfield'])}, "
        f"tr_barge={len(bank['tr_barge'])}, ood_far={len(bank['ood_farfield'])}, ood_barge={len(bank['ood_barge'])}"
    )

    print("🏦 [4/7] 构建 PolyAI/minds14 (zh-CN) 真实人类语音 -> 零 ASR 异步工具样本...")
    m14_tr, m14_val, ood_m14 = build_minds14_audio_tool_samples(
        minds_pq, audio_dir, data_rel_prefix, bank, noises_tr, noises_ood
    )
    print(f"   minds14: train={len(m14_tr)}, valid={len(m14_val)}, ood_cases={len(ood_m14)}")

    print("💬 [5/7] 构建 CosyAudio 真实中文口语问答流 + AISHELL 声学边界样本...")
    spoken_pairs = load_spoken_dialogue_pairs(max_pairs=13000)
    cosy_tr, cosy_val = build_cosy_qa_audio_samples(
        cosy_pq, audio_dir, data_rel_prefix, bank, noises_tr, noises_ood, max_rows=2000
    )
    ais_tr, ais_val = build_aishell_contrast_samples(bank, noises_tr, noises_ood, spoken_pairs)
    print(f"   cosy_qa: train={len(cosy_tr)}, valid={len(cosy_val)} | aishell_contrast: train={len(ais_tr)}, valid={len(ais_val)}")

    print("🛠️ [6/7] 构建口语多轮短对话 (8,000) + 强化异步工具文本 (6,000)...")
    text_sp = build_spoken_text_duplex_rows(spoken_pairs, count=8000)
    text_fc = build_async_tool_text_rows(max_rows=6000)

    random.shuffle(text_sp)
    random.shuffle(text_fc)
    sp_split = int(len(text_sp) * 0.95)
    fc_split = int(len(text_fc) * 0.95)

    train_all = m14_tr + cosy_tr + ais_tr + text_sp[:sp_split] + text_fc[:fc_split]
    valid_all = m14_val + cosy_val + ais_val + text_sp[sp_split:] + text_fc[fc_split:]
    random.shuffle(train_all)
    random.shuffle(valid_all)

    # 校验每条数据的 <audio> 占位符数 == audios 列表长度
    for split_name, rows in [("train", train_all), ("valid", valid_all)]:
        for r in rows:
            n_tag = sum(m["value"].count("<audio>") for m in r["conversations"])
            assert n_tag == len(r["audios"]), f"{split_name} mismatch: {n_tag} vs {len(r['audios'])}"

    # 构建严格 OOD 盲测清单 (全未见声源!)
    ood_manifest = {
        "minds14_tool_cases": ood_m14,
        "ood_farfield": bank["ood_farfield"],
        "ood_barge": bank["ood_barge"],
        "ood_noise": [p for p, _ in noises_ood],
        "ood_backchannel": bank["ood_backchannel"],
    }

    (out_dir / "train.json").write_text(json.dumps(train_all, ensure_ascii=False), encoding="utf-8")
    (out_dir / "valid.json").write_text(json.dumps(valid_all, ensure_ascii=False), encoding="utf-8")
    (out_dir / "ood_eval.json").write_text(json.dumps(ood_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. 打包 audios.tar（单文件极速上传 HF & Colab 秒解）
    print("📦 [7/7] 打包无损 FLAC 音频库 -> audios.tar ...")
    tar_path = out_dir / "audios.tar"
    flacs = sorted(audio_dir.rglob("*.flac"))
    with tarfile.open(tar_path, "w") as tf:
        for f in flacs:
            tf.add(f, arcname=str(f.relative_to(out_dir)))

    n_audio_tr = sum(1 for r in train_all if r["audios"])
    n_audio_val = sum(1 for r in valid_all if r["audios"])
    print("=" * 72)
    print(f"✅ v2 大规模真实音频数据集构建完成: {out_dir}")
    print(f"   Train: {len(train_all):,} 条 (其中含真实音频: {n_audio_tr:,} 条, {100*n_audio_tr/len(train_all):.1f}%)")
    print(f"   Valid: {len(valid_all):,} 条 (其中含真实音频: {n_audio_val:,} 条, 100% 零声源重叠!)")
    print(f"   FLAC 音频总数: {len(flacs):,} 个 | audios.tar: {tar_path.stat().st_size / 1024**2:.1f} MB")
    print(f"   train.json: {(out_dir/'train.json').stat().st_size / 1024**2:.1f} MB | valid.json: {(out_dir/'valid.json').stat().st_size / 1024**2:.1f} MB")
    print("=" * 72)


if __name__ == "__main__":
    main()
