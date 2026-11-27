#!/usr/bin/env python3
"""
Gemma-4-E2B Route 2 (II-A) Pure Anti-Poisoning Stage 1 + Stage 2 Dataset Pipeline.

Stage 1 (Thinker Full-Duplex + Zero-ASR Async Tool-Calling + Intelligence-Preserving Spoken Chat):
  - Category A (Real Human Dual-Track Turn-Taking from MagicDataTech, 360 trajectories):
      * A1: Multi-chunk human speech + real mid-sentence hesitation pause hold (<unused1>)
      * A2: Real [ENS] noise, [SONANT] breath/cough, overlapping "嗯/对/是的" backchannel,
            AudioSet noise, MIT RIR far-field & self-echo ignore (<unused5>) with <|mask_turn|> prefix
      * A3: Real human substantive overlapping interruption / barge-in (<unused6><unused1>) with <|mask_turn|> prefix
  - Category B (Self-Distilled Untouched Gemma-4-E2B Spoken Chat + 50% Tool-Irrelevance Negatives, 420 trajectories):
      * B1: 160 diverse spoken greetings ("你好", "在吗"), identity ("你叫什么名字"), emotions & daily small talk
      * B2: 260 real spoken Chinese prompts from JerryAGENDD/chinese_speech_cosy_audio
      * Answers distilled directly from untouched google/gemma-4-E2B-it in crisp 1~2 sentence natural spoken Mandarin
      * Strictly 50% carry full 9-tool schema in `tools` with ZERO tool calls (curing false tool triggers!)
  - Category C (Multi-Domain Zero-ASR Async Tool-Calling & Mid-Flight Cancellation, 320 trajectories):
      * C1: Real human PolyAI/minds14 (zh-CN) banking/card/bill voice queries sliced at 0.48s frames
      * C2: Multi-speaker neural cross-domain queries across all 9 live tools sliced at 0.48s frames
      * C3: Mid-flight async tool cancellation (<unused6><unused7>c1)

Stage 2 (Talker 1536-dim Hidden States -> CosyVoice2 25Hz Discrete Speech Units):
  - Strictly "Hear 100 voices, speak 1 voice": synthesizes all clean assistant spoken turns using ONE single
    warm Mandarin female voice (zh-CN-XiaoxiaoNeural) + extracts exact 25Hz FSQ units via speech_tokenizer_v2.onnx.
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import av
import edge_tts
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gemma4_omni_route2.vocoder import SpeechTokenizerV2, download_cosy2_assets  # noqa: E402


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PARQUET_CACHE = Path.home() / ".cache" / "gemma4_duplex_v2_parquets"
TTS_AUDIO_CACHE = Path.home() / ".cache" / "gemma4_route2_tts_cache"
DISTILL_CACHE_PATH = Path.home() / ".cache" / "gemma4_route2_self_distill_cache.json"

SR = 16000
CHUNK_SAMPLES = 7680  # 0.48s @ 16kHz physical microphone frame

# Multi-speaker human-like neural voices ONLY for user input ("hear 100 voices")
USER_NEURAL_VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-liaoning-XiaobeiNeural",
    "zh-CN-shaanxi-XiaoniNeural",
    "zh-TW-HsiaoChenNeural",
    "zh-HK-HiuMaanNeural",
]

# Strictly ONE consistent pleasant female voice for ALL Stage 2 assistant target units ("speak 1 voice")
ASSISTANT_SINGLE_VOICE = "zh-CN-XiaoxiaoNeural"

LIVE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "查询指定城市和日期的天气预报",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "set_alarm",
        "description": "设置闹钟或日程语音提醒",
        "parameters": {
            "type": "object",
            "properties": {"time": {"type": "string"}, "label": {"type": "string"}},
            "required": ["time"],
        },
    },
    {
        "name": "control_smart_home",
        "description": "控制家中空调、灯光、窗帘或扫地机器人等智能设备",
        "parameters": {
            "type": "object",
            "properties": {"room": {"type": "string"}, "device": {"type": "string"}, "action": {"type": "string"}},
            "required": ["device", "action"],
        },
    },
    {
        "name": "search_flight",
        "description": "查询两座城市之间的航班与票价信息",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "navigate_to",
        "description": "规划驾车或公交导航路线并估算耗时",
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["destination"],
        },
    },
    {
        "name": "play_music",
        "description": "按歌手、曲风或心情播放音乐",
        "parameters": {
            "type": "object",
            "properties": {"artist_or_mood": {"type": "string"}},
            "required": ["artist_or_mood"],
        },
    },
    {
        "name": "track_package",
        "description": "查询快递物流最新配送进度",
        "parameters": {
            "type": "object",
            "properties": {"courier": {"type": "string"}},
            "required": ["courier"],
        },
    },
    {
        "name": "translate_text",
        "description": "将口语内容翻译为指定目标语言",
        "parameters": {
            "type": "object",
            "properties": {"target_lang": {"type": "string"}, "phrase": {"type": "string"}},
            "required": ["target_lang", "phrase"],
        },
    },
    {
        "name": "check_account_balance",
        "description": "查询银行卡余额、账单明细或卡片状态",
        "parameters": {
            "type": "object",
            "properties": {"query_type": {"type": "string"}},
            "required": ["query_type"],
        },
    },
]

ALL_TOOLS_JSON = json.dumps(LIVE_TOOLS_SCHEMA, ensure_ascii=False)

DAILY_CHAT_PROMPTS: List[str] = [
    # Greetings & Mic Checks
    "你好", "你好呀", "哈喽", "嗨，在吗", "喂喂，听得到我说话吗", "早上好呀", "下午好", "晚上好",
    "嗨你好，今天过得怎么样", "在不在呀", "哈喽，出来聊聊天", "喂，你好",
    # Assistant Identity & Capabilities
    "你叫什么名字", "你是谁呀", "自我介绍一下吧", "你能帮我做什么呀", "你会说中文吗",
    "你是真人还是人工智能助手", "你有什么本领", "我可以怎么称呼你",
    # Gratitude, Politeness & Closing
    "太感谢你了", "谢谢你帮我解答", "辛苦你啦", "你真聪明", "说得太清楚了，谢谢",
    "好的我知道了，先这样吧", "晚安，我要准备睡觉了", "拜拜，下次再聊",
    # Everyday Emotions & Companionship
    "今天上班好累啊，感觉快撑不住了", "周末一个人在家好无聊，不知道干什么好",
    "晚上翻来覆去睡不着怎么办", "今天面试有点紧张，怎么缓解一下",
    "刚才开会出了一点小差错，心情有点低落", "最近总是久坐，肩膀和脖子特别酸",
    "终于把这周的工作忙完了，太开心了", "外面一直在下雨，心情都有点闷闷的",
    # Casual Life, Food, Health & Fun
    "给我讲一个简短好笑的冷笑话吧", "晚上想自己做顿简单的晚饭，有什么快手菜推荐吗",
    "平时喝咖啡好还是喝茶更好一点", "每天跑三公里对身体有什么好处",
    "怎么快速挑到一个又甜又好吃的西瓜", "煮饺子的时候怎么防止饺子皮破掉",
    "夏天衣服上沾了油渍怎么洗干净", "平时怎么养成早睡早起的好习惯",
    "看书总是容易走神怎么办", "养猫和养狗哪个更适合上班族",
    "去海边玩一般需要提前准备些什么东西", "怎么判断自己是不是吃太多盐了",
    "早上起床空腹喝一杯温水有用吗", "感冒嗓子疼的时候吃点什么会舒服些",
    "白衬衫发黄了有什么简单的清洗办法", "学一门新外语最有效的入门方法是什么",
    "久看电脑眼睛干涩怎么缓解", "家里养什么绿植最好养活不容易死",
    "做番茄炒蛋是先放鸡蛋还是先放番茄", "运动完肌肉酸痛应该热敷还是冷敷",
    "怎么克服拖延症把事情按时做完", "坐长途飞机怎么让脖子舒服一点",
    "平时存不下钱有什么实用的记账小技巧", "第一次去露营最容易漏带什么东西",
]

FORBIDDEN_POISON_SUBSTRINGS = [
    "我听到了", "您刚才说", "你刚才说", "小明", "作为AI", "作为一个人工智能",
    "用户打招呼", "用户问", "用户在问", "我应该", "/thought", "thought",
    "**", "##", "```", "1.", "2.", "- ", "<|", "|>",
]


def format_gemma4_call(fn_name: str, args: Dict[str, str]) -> str:
    kv = ",".join(f'{k}:<|"|>{v}<|"|>' for k, v in args.items())
    return f"<|tool_call>call:{fn_name}{{{kv}}}<tool_call|>"


def decode_mp3_bytes_to_16k(mp3_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    container = av.open(io.BytesIO(mp3_bytes))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    chunks: List[np.ndarray] = []
    for frame in container.decode(audio=0):
        for out_f in resampler.resample(frame):
            chunks.append(out_f.to_ndarray()[0])
    for out_f in resampler.resample(None):
        chunks.append(out_f.to_ndarray()[0])
    if not chunks:
        return np.zeros(target_sr, dtype=np.float32)
    wav = np.concatenate(chunks).astype(np.float32)
    return wav


def decode_audio_bytes_to_16k(raw_bytes: bytes) -> np.ndarray:
    try:
        wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != SR:
            g = np.gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        return wav
    except Exception:
        return decode_mp3_bytes_to_16k(raw_bytes, target_sr=SR)


def trim_silence_edges(wav: np.ndarray, thresh: float = 0.008) -> np.ndarray:
    if len(wav) < 800:
        return wav
    win = 400
    n_frames = len(wav) // win
    if n_frames <= 2:
        return wav
    rms = np.sqrt(np.mean(wav[: n_frames * win].reshape(n_frames, win) ** 2, axis=1))
    idx = np.where(rms > thresh)[0]
    if len(idx) == 0:
        return wav
    s = max(0, (idx[0] - 1) * win)
    e = min(len(wav), (idx[-1] + 2) * win)
    return wav[s:e]


def normalize_rms(wav: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    cur = float(np.sqrt(np.mean(wav**2)) + 1e-8)
    out = wav * (target_rms / cur)
    peak = float(np.max(np.abs(out)) + 1e-8)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def slice_into_048s_frames(wav: np.ndarray, max_frames: int = 8) -> List[np.ndarray]:
    """Slice 16kHz speech strictly into 0.48s (7,680-sample) physical microphone frames."""
    wav = trim_silence_edges(wav)
    if len(wav) < 1600:
        wav = np.pad(wav, (0, max(0, CHUNK_SAMPLES - len(wav))))
    wav = normalize_rms(wav, target_rms=random.uniform(0.06, 0.11))

    frames: List[np.ndarray] = []
    for start in range(0, len(wav), CHUNK_SAMPLES):
        seg = wav[start : start + CHUNK_SAMPLES]
        if start > 0 and len(seg) < int(CHUNK_SAMPLES * 0.22):
            break
        if len(seg) < CHUNK_SAMPLES:
            seg = np.pad(seg, (0, CHUNK_SAMPLES - len(seg)))
        frames.append(seg.astype(np.float32))
        if len(frames) >= max_frames:
            break
    if not frames:
        frames = [np.zeros(CHUNK_SAMPLES, dtype=np.float32)]
    return frames


async def _synth_batch_edge_tts(items: List[Tuple[str, str, str]], target_sr: int = 16000) -> Dict[str, Path]:
    """Asynchronously synthesize (cache_key, text, voice) with disk caching."""
    TTS_AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(20)
    results: Dict[str, Path] = {}

    async def _worker(key: str, text: str, voice: str, rate: str) -> None:
        out_flac = TTS_AUDIO_CACHE / f"{key}_{target_sr}.flac"
        if out_flac.exists() and out_flac.stat().st_size > 500:
            results[key] = out_flac
            return
        async with sem:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    buf = bytearray()
                    async for msg in comm.stream():
                        if msg["type"] == "audio":
                            buf.extend(msg["data"])
                    if len(buf) > 200:
                        wav = decode_mp3_bytes_to_16k(bytes(buf), target_sr=target_sr)
                        wav = trim_silence_edges(wav, thresh=0.005)
                        sf.write(str(out_flac), wav, target_sr, format="FLAC")
                        results[key] = out_flac
                        return
                except Exception:
                    await asyncio.sleep(0.4 * (attempt + 1))

    tasks = []
    for key, text, voice in items:
        rate = "+0%" if voice == ASSISTANT_SINGLE_VOICE else random.choice(["-4%", "+0%", "+5%"])
        tasks.append(_worker(key, text, voice, rate))
    await asyncio.gather(*tasks)
    return results


def sanitize_spoken_reply(raw: str) -> Optional[str]:
    text = raw.strip()
    text = re.sub(r"<\|thought>.*?<thought\|>", "", text, flags=re.S).strip()
    text = re.sub(r"[*#`~\[\]()（）<>]", "", text)
    text = re.sub(r"^\s*[\d一二三四五]+[\.、\s]+", "", text)
    text = re.sub(r"\s+", "，", text).strip("， ")
    if not text:
        return None
    if text[-1] not in "。！？":
        text += "。"
    # Keep first 1~2 crisp spoken sentences (10..68 chars)
    sents = re.findall(r"[^。！？]+[。！？]", text)
    if sents:
        acc = ""
        for s in sents:
            if len(acc) + len(s) <= 64 or not acc:
                acc += s
            else:
                break
        text = acc[:68]
        if text[-1] not in "。！？":
            text += "。"
    if len(text) < 6 or len(text) > 72:
        return None
    for bad in FORBIDDEN_POISON_SUBSTRINGS:
        if bad in text:
            return None
    return text


def run_gemma4_self_distillation(prompts: List[str]) -> Dict[str, str]:
    """Generate crisp 1~2 sentence natural spoken Chinese answers from UNTOUCHED google/gemma-4-E2B-it."""
    cached: Dict[str, str] = {}
    if DISTILL_CACHE_PATH.exists():
        try:
            cached = json.loads(DISTILL_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cached = {}

    missing = [p for p in prompts if p not in cached or sanitize_spoken_reply(cached[p]) is None]
    if not missing:
        print(f"✅ Loaded all {len(prompts)} self-distilled Gemma-4-E2B spoken replies from cache.")
        return {p: sanitize_spoken_reply(cached[p]) or cached[p] for p in prompts if sanitize_spoken_reply(cached.get(p, ""))}

    print(f"🧠 Self-distilling {len(missing)} natural spoken replies from untouched google/gemma-4-E2B-it...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    tok.padding_side = "left"
    model = Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-E2B-it", dtype=dtype).to(device).eval()

    sys_inst = (
        "你是一个温和、自然、口语化的中文全双工语音助手。"
        "请直接用一到两句简短自然的中文口语回答用户，总字数控制在15到45个字以内，"
        "绝对不要使用任何Markdown符号、星号、列表编号或括号。"
    )
    bsz = 12
    for i in range(0, len(missing), bsz):
        batch_p = missing[i : i + bsz]
        formatted = [
            tok.apply_chat_template(
                [{"role": "system", "content": sys_inst}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in batch_p
        ]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=56,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<turn|>")],
            )
        in_len = enc["input_ids"].shape[1]
        for q, out_row in zip(batch_p, gen):
            raw = tok.decode(out_row[in_len:], skip_special_tokens=False)
            raw = re.sub(r"<\|channel>.*?<channel\|>", "", raw, flags=re.S)
            raw = raw.replace("<turn|>", "").replace("<eos>", "").replace("<pad>", "").strip()
            clean = sanitize_spoken_reply(raw)
            if clean:
                cached[q] = clean
        if (i // bsz) % 5 == 0 or (i + bsz) >= len(missing):
            print(f"   Distilled {min(i + bsz, len(missing))}/{len(missing)}...")
            DISTILL_CACHE_PATH.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return {p: sanitize_spoken_reply(cached[p]) for p in prompts if p in cached and sanitize_spoken_reply(cached[p])}


def load_noise_and_rir_pools() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    noise_pool: List[np.ndarray] = []
    rir_pool: List[np.ndarray] = []

    as_parquets = glob.glob(str(HF_CACHE / "datasets--agkphysics--AudioSet" / "**" / "*.parquet"), recursive=True)
    for pf in as_parquets[:2]:
        try:
            tbl = pq.read_table(pf, columns=["audio"])
            for i in range(min(120, len(tbl))):
                b = tbl["audio"][i].as_py()["bytes"]
                w = decode_audio_bytes_to_16k(b)
                if len(w) >= CHUNK_SAMPLES:
                    noise_pool.append(w[: CHUNK_SAMPLES * 2])
        except Exception:
            pass

    rir_parquets = glob.glob(
        str(HF_CACHE / "datasets--davidscripka--MIT_environmental_impulse_responses" / "**" / "*.parquet"),
        recursive=True,
    )
    for pf in rir_parquets[:1]:
        try:
            tbl = pq.read_table(pf, columns=["audio"])
            for i in range(min(80, len(tbl))):
                b = tbl["audio"][i].as_py()["bytes"]
                w = decode_audio_bytes_to_16k(b)
                if len(w) > 200:
                    w = w / (np.max(np.abs(w)) + 1e-8)
                    rir_pool.append(w[:8000])
        except Exception:
            pass
    return noise_pool, rir_pool


def make_farfield_or_echo(wav: np.ndarray, rir_pool: List[np.ndarray], is_echo: bool = False) -> np.ndarray:
    out = wav.copy()
    if is_echo:
        sos = butter(4, [300.0 / (SR / 2), 3400.0 / (SR / 2)], btype="bandpass", output="sos")
        out = sosfilt(sos, out).astype(np.float32)
    if rir_pool:
        rir = random.choice(rir_pool)
        out = fftconvolve(out, rir, mode="full")[: len(wav)].astype(np.float32)
    target_rms = random.uniform(0.012, 0.028) if is_echo else random.uniform(0.015, 0.032)
    return normalize_rms(out, target_rms=target_rms)


def parse_magicdata_conversations() -> Dict[str, List[Tuple[str, Path, List[Tuple[float, float, str, str]]]]]:
    """Parse all 27 dual-track real human spontaneous Mandarin conversations from MagicDataTech."""
    md_snaps = glob.glob(
        str(
            HF_CACHE
            / "datasets--MagicDataTech--multi-stream-spontaneous-conversation-training-datasets_chinese"
            / "snapshots"
            / "*"
        )
    )
    if not md_snaps:
        raise FileNotFoundError("MagicDataTech dataset not found in HF cache.")
    md_root = Path(md_snaps[0])
    groups: Dict[str, List[Tuple[str, Path, List[Tuple[float, float, str, str]]]]] = {}

    line_re = re.compile(r"\[(\d+\.\d+),(\d+\.\d+)\]\s+(\S+)\s+\S+\s+(.*)")
    for txt_p in sorted((md_root / "TXT").glob("*.txt")):
        stem = txt_p.stem  # e.g. A1012_S0004_0_G0019
        conv_id = "_".join(stem.split("_")[:3])
        spk_id = stem.split("_")[-1]
        wav_p = md_root / "WAV" / f"{stem}.wav"
        if not wav_p.exists():
            continue
        segs: List[Tuple[float, float, str, str]] = []
        for raw_l in txt_p.read_text(encoding="utf-8-sig").splitlines():
            m = line_re.match(raw_l.strip())
            if m:
                segs.append((float(m.group(1)), float(m.group(2)), m.group(3), m.group(4).strip()))
        groups.setdefault(conv_id, []).append((spk_id, wav_p, segs))
    return groups


def build_tool_query_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    rng = random.Random(42)

    cities = ["北京", "上海", "深圳", "杭州", "成都", "广州", "武汉", "南京", "西安", "重庆", "苏州", "青岛"]
    dates = ["今天", "明天", "后天", "这周末"]
    for _ in range(30):
        c, d = rng.choice(cities), rng.choice(dates)
        q = rng.choice([f"帮我查一下{d}{c}的天气怎么样", f"看看{c}{d}会不会下雨", f"查一下{c}{d}的气温是多少"])
        specs.append({
            "query": q,
            "fn": "get_weather",
            "args": {"city": c, "date": d},
            "ack": f"好的，我马上帮您查一下{c}{d}的天气。",
            "obs": json.dumps({"city": c, "date": d, "weather": "多云转晴", "temp": "19到26度", "aqi": 38}, ensure_ascii=False),
            "res": f"查到了，{c}{d}多云转晴，气温十九到二十六度，空气质量很不错。",
        })

    times = ["明天早上七点", "早上六点半", "下午三点", "晚上九点半", "二十分钟后"]
    labels = ["起床", "开项目会", "去接孩子", "吃药", "健身"]
    for _ in range(25):
        t, lb = rng.choice(times), rng.choice(labels)
        q = rng.choice([f"帮我定一个{t}{lb}的闹钟", f"设置{t}提醒我{lb}"])
        specs.append({
            "query": q,
            "fn": "set_alarm",
            "args": {"time": t, "label": lb},
            "ack": f"没问题，正在为您设置{t}的提醒。",
            "obs": json.dumps({"ok": True, "time": t, "label": lb}, ensure_ascii=False),
            "res": f"已经帮您定好{t}{lb}的闹钟了。",
        })

    rooms = ["客厅", "卧室", "书房", "儿童房"]
    devs = [("空调", "打开"), ("空调", "调到二十四度"), ("吸顶灯", "关掉"), ("窗帘", "拉开"), ("扫地机器人", "开始清扫")]
    for _ in range(28):
        rm, (dv, act) = rng.choice(rooms), rng.choice(devs)
        q = rng.choice([f"帮我把{rm}的{dv}{act}", f"{act}{rm}里的{dv}"])
        specs.append({
            "query": q,
            "fn": "control_smart_home",
            "args": {"room": rm, "device": dv, "action": act},
            "ack": f"好的，正在为您{act}{rm}的{dv}。",
            "obs": json.dumps({"ok": True, "room": rm, "device": dv, "action": act}, ensure_ascii=False),
            "res": f"已经帮您把{rm}的{dv}{act}了。",
        })

    pairs = [("北京", "三亚"), ("上海", "成都"), ("深圳", "西安"), ("广州", "杭州"), ("成都", "厦门")]
    for _ in range(24):
        org, dst = rng.choice(pairs)
        d = rng.choice(["明天", "后天", "这周五"])
        q = f"帮我查一下{d}从{org}飞{dst}的机票"
        specs.append({
            "query": q,
            "fn": "search_flight",
            "args": {"origin": org, "destination": dst, "date": d},
            "ack": f"稍等，我帮您查一下{d}{org}飞往{dst}的航班。",
            "obs": json.dumps({"origin": org, "destination": dst, "flights": [{"no": "CA1502", "time": "09:30", "price": 760}]}, ensure_ascii=False),
            "res": f"查到了，{d}{org}飞{dst}最早一班是早上九点半起飞，票价七百六十元。",
        })

    places = ["虹桥火车站", "首都国际机场", "西湖景区", "市中心医院", "科技园地铁站"]
    modes = ["开车", "坐地铁", "打车"]
    for _ in range(24):
        pl, md = rng.choice(places), rng.choice(modes)
        q = f"帮我导航去{pl}，看看{md}要多久"
        specs.append({
            "query": q,
            "fn": "navigate_to",
            "args": {"destination": pl, "mode": md},
            "ack": f"好的，正在为您规划去{pl}的路线。",
            "obs": json.dumps({"destination": pl, "mode": md, "duration": "26分钟", "traffic": "畅通"}, ensure_ascii=False),
            "res": f"路线规划好了，{md}去{pl}预计需要二十六分钟，目前路况畅通。",
        })

    moods = ["轻柔的纯音乐", "周杰伦的歌", "适合跑步的动感音乐", "睡前助眠钢琴曲", "轻松的爵士乐"]
    for _ in range(22):
        m = rng.choice(moods)
        q = rng.choice([f"帮我放一点{m}", f"我想听{m}，帮我播放吧"])
        specs.append({
            "query": q,
            "fn": "play_music",
            "args": {"artist_or_mood": m},
            "ack": f"好嘞，马上为您播放{m}。",
            "obs": json.dumps({"playing": True, "artist_or_mood": m}, ensure_ascii=False),
            "res": f"已经开始为您播放{m}了，希望您喜欢。",
        })

    couriers = ["顺丰", "京东", "中通", "圆通", "韵达"]
    for _ in range(20):
        cr = rng.choice(couriers)
        q = f"帮我查一下我的{cr}快递到哪儿了"
        specs.append({
            "query": q,
            "fn": "track_package",
            "args": {"courier": cr},
            "ack": f"好的，我帮您查一下{cr}快递的最新物流。",
            "obs": json.dumps({"courier": cr, "status": "派送中", "eta": "今天下午四点前"}, ensure_ascii=False),
            "res": f"您的{cr}快递正在派送中，预计今天下午四点前送达。",
        })

    phrases = [
        ("英语", "请问地铁站在哪里", "Excuse me, where is the subway station?"),
        ("英语", "很高兴认识你", "Nice to meet you."),
        ("日语", "谢谢你的帮助", "助けてくれてありがとうございます。"),
        ("英语", "我们可以开始开会了吗", "Shall we start the meeting now?"),
    ]
    for _ in range(20):
        lang, ph, tr = rng.choice(phrases)
        q = f"帮我把{ph}翻译成{lang}"
        specs.append({
            "query": q,
            "fn": "translate_text",
            "args": {"target_lang": lang, "phrase": ph},
            "ack": f"好的，马上帮您翻译成{lang}。",
            "obs": json.dumps({"target_lang": lang, "result": tr}, ensure_ascii=False),
            "res": f"这句话用{lang}说是：{tr}",
        })

    return specs


def main() -> None:
    random.seed(2026)
    np.random.seed(2026)

    s1_dir = REPO_ROOT / "data" / "gemma4_route2_stage1"
    s1_audio_dir = s1_dir / "audios"
    s1_audio_dir.mkdir(parents=True, exist_ok=True)

    s2_dir = REPO_ROOT / "data" / "gemma4_route2_stage2"
    s2_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample 260 real spoken Chinese questions from cosy_qa_0 + cosy_qa_1
    cosy_rows: List[Tuple[str, bytes]] = []
    for pf_name in ("cosy_qa_0.parquet", "cosy_qa_1.parquet"):
        pf = PARQUET_CACHE / pf_name
        if not pf.exists():
            continue
        tbl = pq.read_table(pf, columns=["transcript", "audio"])
        indices = list(range(len(tbl)))
        random.shuffle(indices)
        for idx in indices:
            q_txt = str(tbl["transcript"][idx].as_py()).strip()
            if 5 <= len(q_txt) <= 36 and not re.search(r"[A-Za-z_/{}\[\]]", q_txt):
                b = tbl["audio"][idx].as_py()["bytes"]
                cosy_rows.append((q_txt, b))
                if len(cosy_rows) >= 280:
                    break
        if len(cosy_rows) >= 280:
            break
    cosy_rows = cosy_rows[:260]

    # Expand daily prompts to 160 items (allowing multi-speaker variations of greetings!)
    daily_expanded: List[str] = []
    high_freq_greetings = DAILY_CHAT_PROMPTS[:20]  # extra weight on "你好", "在吗", "你叫什么名字"!
    daily_expanded.extend(high_freq_greetings * 4)  # 80 items
    daily_expanded.extend(DAILY_CHAT_PROMPTS[20:] * 2)  # ~80 items
    daily_expanded = daily_expanded[:160]

    all_distill_prompts = list(dict.fromkeys(DAILY_CHAT_PROMPTS + [q for q, _ in cosy_rows]))
    distilled_map = run_gemma4_self_distillation(all_distill_prompts)

    # 2. Synthesize user neural audio (B1 + C2 + C3) AND single-voice Stage 2 assistant audio!
    tool_specs = build_tool_query_specs()
    cancel_phrases = [
        "等一下，先不用查了", "算了算了，取消刚才那个吧", "先停一下不用办了",
        "不好意思，刚才说错了先取消吧", "不用查这个了，先停下",
    ]

    tts_jobs: List[Tuple[str, str, str]] = []
    # Reference clean voice clip for Cosy2StreamingVocoder zero-shot timbre lock!
    ref_text = "你好呀！我是你的全双工语音助手，随时可以陪你聊天或者帮你查询信息。"
    tts_jobs.append(("ref_xiaoxiao_prompt", ref_text, ASSISTANT_SINGLE_VOICE))

    b1_user_keys: List[Tuple[str, str, str]] = []
    for idx, q_txt in enumerate(daily_expanded):
        if q_txt not in distilled_map:
            continue
        v = USER_NEURAL_VOICES[idx % len(USER_NEURAL_VOICES)]
        k = f"u_b1_{idx:04d}_" + hashlib.md5(f"{q_txt}_{v}_{idx}".encode()).hexdigest()[:8]
        tts_jobs.append((k, q_txt, v))
        b1_user_keys.append((k, q_txt, distilled_map[q_txt]))

    c2_keys: List[Tuple[str, Dict[str, Any]]] = []
    for idx, sp in enumerate(tool_specs):
        v = USER_NEURAL_VOICES[idx % len(USER_NEURAL_VOICES)]
        k = f"u_c2_{idx:04d}_" + hashlib.md5(f"{sp['query']}_{v}".encode()).hexdigest()[:8]
        tts_jobs.append((k, sp["query"], v))
        c2_keys.append((k, sp))

    c3_cancel_keys: List[str] = []
    for idx in range(80):
        cp = cancel_phrases[idx % len(cancel_phrases)]
        v = USER_NEURAL_VOICES[idx % len(USER_NEURAL_VOICES)]
        k = f"u_c3_{idx:04d}_" + hashlib.md5(f"{cp}_{v}_{idx}".encode()).hexdigest()[:8]
        tts_jobs.append((k, cp, v))
        c3_cancel_keys.append(k)

    # Collect ALL unique spoken assistant texts for Stage 2 single-voice synthesis!
    minds14_replies = [
        ("好的，我马上帮您查一下账户和账单情况。", "已经帮您查到了，您的账户状态正常，本期账单没有逾期记录。"),
        ("稍等，我这就为您核实卡片和余额信息。", "查到了，您的银行卡目前状态正常，活期可用余额充足。"),
        ("没问题，正在为您查询最新的账户明细。", "已经核实好了，最近一笔交易记录正常，请您放心。"),
    ]
    unique_assistant_texts: Dict[str, str] = {}
    for ans in distilled_map.values():
        ak = "a_" + hashlib.md5(ans.encode()).hexdigest()[:12]
        unique_assistant_texts[ans] = ak
    for sp in tool_specs:
        for t in (sp["ack"], sp["res"]):
            unique_assistant_texts[t] = "a_" + hashlib.md5(t.encode()).hexdigest()[:12]
    for ack_t, res_t in minds14_replies:
        for t in (ack_t, res_t):
            unique_assistant_texts[t] = "a_" + hashlib.md5(t.encode()).hexdigest()[:12]

    for a_txt, a_key in unique_assistant_texts.items():
        tts_jobs.append((a_key, a_txt, ASSISTANT_SINGLE_VOICE))

    print(f"🎙️ Synthesizing {len(tts_jobs)} neural audio clips (cached incremental)...")
    synth_paths = asyncio.run(_synth_batch_edge_tts(tts_jobs, target_sr=SR))

    # Save studio 24kHz reference prompt for Cosy2StreamingVocoder!
    ref_24k_jobs = [("ref_xiaoxiao_24k", ref_text, ASSISTANT_SINGLE_VOICE)]
    ref_24k_map = asyncio.run(_synth_batch_edge_tts(ref_24k_jobs, target_sr=24000))
    if "ref_xiaoxiao_24k" in ref_24k_map:
        ref_dst = REPO_ROOT / "src" / "gemma4_omni_route2" / "assets" / "prompt_zh.wav"
        wav24, _ = sf.read(str(ref_24k_map["ref_xiaoxiao_24k"]), dtype="float32")
        sf.write(str(ref_dst), wav24[: 24000 * 3], 24000)
        print(f"✅ Updated single-speaker reference prompt: {ref_dst} ({len(wav24[:24000*3])/24000:.2f}s @ 24kHz)")

    # Extract 25Hz CosyVoice2 discrete speech units for ALL Stage 2 assistant utterances (with disk cache)!
    units_cache_p = Path.home() / ".cache" / "gemma4_route2_units_cache.json"
    units_cache: Dict[str, List[int]] = {}
    if units_cache_p.exists():
        try:
            units_cache = json.loads(units_cache_p.read_text(encoding="utf-8"))
        except Exception:
            units_cache = {}

    cosy_dir = download_cosy2_assets()
    s3_tok = SpeechTokenizerV2(os.path.join(cosy_dir, "speech_tokenizer_v2.onnx"), prefer_cuda=False)
    assistant_unit_map: Dict[str, List[int]] = {}
    print(f"🔢 Extracting 25Hz CosyVoice2 discrete speech units for {len(unique_assistant_texts)} assistant texts...")
    for a_txt, a_key in unique_assistant_texts.items():
        if a_key in units_cache and len(units_cache[a_key]) >= 5:
            assistant_unit_map[a_txt] = units_cache[a_key]
            continue
        p = synth_paths.get(a_key)
        if p and p.exists():
            u = s3_tok.extract_units(str(p))
            if len(u) >= 5:
                assistant_unit_map[a_txt] = u
                units_cache[a_key] = u
    units_cache_p.write_text(json.dumps(units_cache), encoding="utf-8")
    print(f"✅ Extracted 25Hz discrete speech units for {len(assistant_unit_map)} assistant texts.")

    stage1_records: List[Dict[str, Any]] = []
    stage2_records: List[Dict[str, Any]] = []
    audio_counter = 0

    def save_chunk_flac(chunk_wav: np.ndarray, prefix: str) -> str:
        nonlocal audio_counter
        fname = f"{prefix}_{audio_counter:06d}.flac"
        audio_counter += 1
        sf.write(str(s1_audio_dir / fname), chunk_wav, SR, format="FLAC")
        return f"gemma4_route2_stage1/audios/{fname}"

    # ---------------- CATEGORY B: Self-Distilled Spoken Chat (50% with Tools -> Zero Tool Call!) ----------------
    cat_b_items: List[Tuple[List[np.ndarray], str, str, str]] = []
    for k, q_txt, ans_txt in b1_user_keys:
        if k not in synth_paths:
            continue
        w, _ = sf.read(str(synth_paths[k]), dtype="float32")
        cat_b_items.append((slice_into_048s_frames(w), q_txt, ans_txt, "B1_daily_chat"))

    for q_txt, raw_b in cosy_rows:
        if q_txt not in distilled_map:
            continue
        w = decode_audio_bytes_to_16k(raw_b)
        cat_b_items.append((slice_into_048s_frames(w), q_txt, distilled_map[q_txt], "B2_real_cosy_qa"))

    for idx, (frames, q_txt, ans_txt, subtag) in enumerate(cat_b_items):
        convs: List[Dict[str, str]] = []
        auds: List[str] = []
        for fr in frames:
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(fr, subtag))
            convs.append({"from": "gpt", "value": "<unused1>"})
        convs.append({"from": "human", "value": "<unused0>"})
        convs.append({"from": "gpt", "value": f"<unused2>{ans_txt}"})

        # Strictly 50% carry ALL 9 tools while making ZERO tool calls!
        tools_val = ALL_TOOLS_JSON if (idx % 2 == 0) else ""
        rec = {
            "conversations": convs,
            "tools": tools_val,
            "audios": auds,
            "_meta": {"category": f"{subtag}_{'with_tools_neg' if tools_val else 'plain'}", "user_text": q_txt},
        }
        stage1_records.append(rec)
        if ans_txt in assistant_unit_map:
            stage2_records.append({
                "conversations": convs,
                "tools": tools_val,
                "audios": auds,
                "spoken_text": ans_txt,
                "speech_units": assistant_unit_map[ans_txt],
                "category": subtag,
            })

    # ---------------- CATEGORY A: Real Human Dual-Track MagicDataTech (Zero Persona Pollution!) ----------------
    noise_pool, rir_pool = load_noise_and_rir_pools()
    md_groups = parse_magicdata_conversations()

    hesitation_spans: List[Tuple[Path, float, float, str]] = []
    backchannel_or_noise_spans: List[Tuple[Path, float, float, str]] = []
    substantive_human_spans: List[Tuple[Path, float, float, str]] = []

    for conv_id, tracks in md_groups.items():
        for spk_id, wav_p, segs in tracks:
            for si, (s0, e0, tag, txt) in enumerate(segs):
                dur = e0 - s0
                if tag == "0" or txt in ("[ENS]", "[SONANT]", "嗯。", "对。", "是。", "啊。", "哦。", "嗯嗯。"):
                    if 0.30 <= dur <= 1.50:
                        backchannel_or_noise_spans.append((wav_p, s0, min(e0, s0 + 0.96), txt))
                elif 1.25 <= dur <= 3.80 and len(txt) >= 6 and "[" not in txt:
                    substantive_human_spans.append((wav_p, s0, e0, txt))
                # Check consecutive segments by same speaker with real mid-sentence hesitation gap (0.20s..1.25s)
                if si + 1 < len(segs):
                    s1, e1, tag1, txt1 = segs[si + 1]
                    gap = s1 - e0
                    if tag != "0" and tag1 != "0" and 0.20 <= gap <= 1.25 and (e0 - s0) >= 0.6 and (e1 - s1) >= 0.6:
                        if "[" not in txt and "[" not in txt1:
                            w_start = max(s0, e0 - 1.44)
                            w_end = min(e1, s1 + 1.44)
                            hesitation_spans.append((wav_p, w_start, w_end, f"{txt} [停顿{gap:.2f}s] {txt1}"))

    random.shuffle(hesitation_spans)
    random.shuffle(backchannel_or_noise_spans)
    random.shuffle(substantive_human_spans)

    clean_partial_prefixes = [
        ans[: max(8, len(ans) // 2)]
        for ans in list(distilled_map.values())
        if len(ans) >= 18
    ]

    # A1: 120 real human multi-chunk speech + mid-sentence hesitation hold (<unused1>)
    wav_cache: Dict[Path, np.ndarray] = {}

    def read_md_slice(wp: Path, t0: float, t1: float) -> np.ndarray:
        if wp not in wav_cache:
            w, sr = sf.read(str(wp), dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=-1)
            wav_cache[wp] = w
        w = wav_cache[wp]
        return w[int(t0 * SR) : int(t1 * SR)].copy()

    for wp, t0, t1, desc in hesitation_spans[:120]:
        seg = read_md_slice(wp, t0, t1)
        frames = slice_into_048s_frames(seg, max_frames=8)
        if len(frames) < 2:
            continue
        convs = []
        auds = []
        for fr in frames:
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(fr, "A1_hesit"))
            convs.append({"from": "gpt", "value": "<unused1>"})
        stage1_records.append({
            "conversations": convs,
            "tools": ALL_TOOLS_JSON if random.random() < 0.5 else "",
            "audios": auds,
            "_meta": {"category": "A1_magicdata_hesitation_hold", "user_text": desc},
        })

    # A2: 120 mid-playback <unused5> ignore (real MagicDataTech [ENS]/[SONANT]/backchannel + AudioSet + far-field + self-echo)
    assistant_echo_wavs = [
        sf.read(str(synth_paths[k]), dtype="float32")[0]
        for k in list(unique_assistant_texts.values())[:40]
        if k in synth_paths
    ]
    for i in range(120):
        prefix = random.choice(clean_partial_prefixes)
        mode = i % 4
        if mode in (0, 1) and backchannel_or_noise_spans:
            wp, t0, t1, label_txt = backchannel_or_noise_spans[i % len(backchannel_or_noise_spans)]
            raw_clip = read_md_slice(wp, t0, t1)
            if len(raw_clip) < CHUNK_SAMPLES:
                raw_clip = np.pad(raw_clip, (0, CHUNK_SAMPLES - len(raw_clip)))
            clip = normalize_rms(raw_clip[:CHUNK_SAMPLES], target_rms=random.uniform(0.03, 0.07))
            sub = f"magicdata_{label_txt}"
        elif mode == 2 and assistant_echo_wavs:
            ew = random.choice(assistant_echo_wavs)
            if len(ew) < CHUNK_SAMPLES:
                ew = np.pad(ew, (0, CHUNK_SAMPLES - len(ew)))
            clip = make_farfield_or_echo(ew[:CHUNK_SAMPLES], rir_pool, is_echo=True)
            sub = "self_echo_residual"
        else:
            if substantive_human_spans and random.random() < 0.5:
                wp, t0, t1, _ = random.choice(substantive_human_spans)
                fw = read_md_slice(wp, t0, min(t1, t0 + 0.48))
                if len(fw) < CHUNK_SAMPLES:
                    fw = np.pad(fw, (0, CHUNK_SAMPLES - len(fw)))
                clip = make_farfield_or_echo(fw[:CHUNK_SAMPLES], rir_pool, is_echo=False)
                sub = "farfield_bystander"
            elif noise_pool:
                nw = random.choice(noise_pool)[:CHUNK_SAMPLES]
                clip = normalize_rms(nw, target_rms=random.uniform(0.025, 0.06))
                sub = "audioset_noise"
            else:
                clip = (np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.015)
                sub = "room_noise"

        # <|mask_turn|> masks the partial assistant prefix (-100) so the model NEVER learns to truncate speech early!
        convs = [
            {"from": "human", "value": "<unused0>"},
            {"from": "gpt", "value": f"<|mask_turn|><unused2>{prefix}"},
            {"from": "human", "value": "<audio>"},
            {"from": "gpt", "value": "<unused5>"},
        ]
        auds = [save_chunk_flac(clip, "A2_ign")]
        stage1_records.append({
            "conversations": convs,
            "tools": ALL_TOOLS_JSON if random.random() < 0.5 else "",
            "audios": auds,
            "_meta": {"category": f"A2_ignore_{sub}", "user_text": sub},
        })

    # A3: 120 real human substantive barge-in interruptions (<unused6><unused1> -> hold <unused1>)
    for i, (wp, t0, t1, h_txt) in enumerate(substantive_human_spans[:120]):
        prefix = random.choice(clean_partial_prefixes)
        seg = read_md_slice(wp, t0, t1)
        if noise_pool and random.random() < 0.35:
            nw = random.choice(noise_pool)
            if len(nw) < len(seg):
                nw = np.tile(nw, int(np.ceil(len(seg) / len(nw))))
            seg = seg + nw[: len(seg)] * 0.18
        frames = slice_into_048s_frames(seg, max_frames=6)

        convs = [
            {"from": "human", "value": "<unused0>"},
            {"from": "gpt", "value": f"<|mask_turn|><unused2>{prefix}"},
        ]
        auds = []
        for fi, fr in enumerate(frames):
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(fr, "A3_barge"))
            # Frame 0 of human barge-in -> immediate <unused6><unused1>; subsequent frames -> hold <unused1>
            convs.append({"from": "gpt", "value": "<unused6><unused1>" if fi == 0 else "<unused1>"})

        stage1_records.append({
            "conversations": convs,
            "tools": ALL_TOOLS_JSON if random.random() < 0.5 else "",
            "audios": auds,
            "_meta": {"category": "A3_magicdata_real_barge_in", "user_text": h_txt},
        })

    # ---------------- CATEGORY C: Clean Async Tool-Calling & Mid-Flight Cancellation ----------------
    # C1: 60 real human PolyAI/minds14 (zh-CN) banking/card/bill voice queries
    m14_path = PARQUET_CACHE / "minds14_zh.parquet"
    if m14_path.exists():
        m_tbl = pq.read_table(m14_path)
        for idx in range(min(60, len(m_tbl))):
            raw_b = m_tbl["audio"][idx].as_py()["bytes"]
            q_txt = str(m_tbl["transcription"][idx].as_py())
            w = decode_audio_bytes_to_16k(raw_b)
            frames = slice_into_048s_frames(w, max_frames=8)
            ack_t, res_t = minds14_replies[idx % len(minds14_replies)]
            call_s = format_gemma4_call("check_account_balance", {"query_type": "账户与账单查询"})
            obs_s = json.dumps({"status": "正常", "overdue": False, "balance_ok": True}, ensure_ascii=False)

            convs = []
            auds = []
            for fr in frames:
                convs.append({"from": "human", "value": "<audio>"})
                auds.append(save_chunk_flac(fr, "C1_m14"))
                convs.append({"from": "gpt", "value": "<unused1>"})
            convs.append({"from": "human", "value": "<unused0>"})
            convs.append({"from": "gpt", "value": f"<unused3>{call_s}{ack_t}"})
            convs.append({"from": "observation", "value": obs_s})
            convs.append({"from": "gpt", "value": f"<unused4>{res_t}"})

            stage1_records.append({
                "conversations": convs,
                "tools": ALL_TOOLS_JSON,
                "audios": auds,
                "_meta": {"category": "C1_minds14_real_tool", "user_text": q_txt},
            })
            if ack_t in assistant_unit_map:
                stage2_records.append({
                    "conversations": convs[:-2],
                    "tools": ALL_TOOLS_JSON,
                    "audios": auds,
                    "spoken_text": ack_t,
                    "speech_units": assistant_unit_map[ack_t],
                    "category": "C1_tool_ack",
                })
            if res_t in assistant_unit_map:
                stage2_records.append({
                    "conversations": convs,
                    "tools": ALL_TOOLS_JSON,
                    "audios": auds,
                    "spoken_text": res_t,
                    "speech_units": assistant_unit_map[res_t],
                    "category": "C1_tool_res",
                })

    # C2: Multi-speaker cross-domain async tool calls across all 9 tools
    for k, sp in c2_keys:
        if k not in synth_paths:
            continue
        w, _ = sf.read(str(synth_paths[k]), dtype="float32")
        frames = slice_into_048s_frames(w, max_frames=8)
        call_s = format_gemma4_call(sp["fn"], sp["args"])

        convs = []
        auds = []
        for fr in frames:
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(fr, "C2_tool"))
            convs.append({"from": "gpt", "value": "<unused1>"})
        convs.append({"from": "human", "value": "<unused0>"})
        convs.append({"from": "gpt", "value": f"<unused3>{call_s}{sp['ack']}"})
        convs.append({"from": "observation", "value": sp["obs"]})
        convs.append({"from": "gpt", "value": f"<unused4>{sp['res']}"})

        stage1_records.append({
            "conversations": convs,
            "tools": ALL_TOOLS_JSON,
            "audios": auds,
            "_meta": {"category": f"C2_tool_{sp['fn']}", "user_text": sp["query"]},
        })
        if sp["ack"] in assistant_unit_map:
            stage2_records.append({
                "conversations": convs[:-2],
                "tools": ALL_TOOLS_JSON,
                "audios": auds,
                "spoken_text": sp["ack"],
                "speech_units": assistant_unit_map[sp["ack"]],
                "category": "C2_tool_ack",
            })
        if sp["res"] in assistant_unit_map:
            stage2_records.append({
                "conversations": convs,
                "tools": ALL_TOOLS_JSON,
                "audios": auds,
                "spoken_text": sp["res"],
                "speech_units": assistant_unit_map[sp["res"]],
                "category": "C2_tool_res",
            })

    # C3: 80 Mid-Flight Async Tool Cancellations (<unused6><unused7>c1)
    cancel_reply = "好的，已经帮您取消刚才的查询了。"
    for idx, ck in enumerate(c3_cancel_keys):
        if ck not in synth_paths or not c2_keys:
            continue
        uk, sp = c2_keys[idx % len(c2_keys)]
        if uk not in synth_paths:
            continue
        uw, _ = sf.read(str(synth_paths[uk]), dtype="float32")
        cw, _ = sf.read(str(synth_paths[ck]), dtype="float32")
        u_frames = slice_into_048s_frames(uw, max_frames=6)
        c_frames = slice_into_048s_frames(cw, max_frames=5)
        call_s = format_gemma4_call(sp["fn"], sp["args"])

        convs = []
        auds = []
        for fr in u_frames:
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(fr, "C3_q"))
            convs.append({"from": "gpt", "value": "<unused1>"})
        convs.append({"from": "human", "value": "<unused0>"})
        convs.append({"from": "gpt", "value": f"<unused3>{call_s}{sp['ack']}"})

        for fi, cfr in enumerate(c_frames):
            convs.append({"from": "human", "value": "<audio>"})
            auds.append(save_chunk_flac(cfr, "C3_cancel"))
            convs.append({"from": "gpt", "value": "<unused6><unused7>c1" if fi == 0 else "<unused1>"})
        convs.append({"from": "human", "value": "<unused0>"})
        convs.append({"from": "gpt", "value": f"<unused2>{cancel_reply}"})

        stage1_records.append({
            "conversations": convs,
            "tools": ALL_TOOLS_JSON,
            "audios": auds,
            "_meta": {"category": "C3_async_tool_cancel", "user_text": sp["query"]},
        })

    # Shuffle and split Train / Valid
    random.shuffle(stage1_records)
    random.shuffle(stage2_records)

    # Save audit file with _meta intact, and clean ShareGPT files without _meta for LlamaFactory
    (s1_dir / "audit_samples.json").write_text(
        json.dumps(stage1_records[:60], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clean_s1 = [
        {"conversations": r["conversations"], "tools": r["tools"], "audios": r["audios"]}
        for r in stage1_records
    ]
    n_val1 = max(60, int(len(clean_s1) * 0.07))
    s1_train, s1_valid = clean_s1[:-n_val1], clean_s1[-n_val1:]
    (s1_dir / "train.json").write_text(json.dumps(s1_train, ensure_ascii=False, indent=2), encoding="utf-8")
    (s1_dir / "valid.json").write_text(json.dumps(s1_valid, ensure_ascii=False, indent=2), encoding="utf-8")

    n_val2 = max(40, int(len(stage2_records) * 0.07))
    s2_train, s2_valid = stage2_records[:-n_val2], stage2_records[-n_val2:]
    with open(s2_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in s2_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(s2_dir / "valid.jsonl", "w", encoding="utf-8") as f:
        for r in s2_valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Register in data/dataset_info.json
    ds_info_p = REPO_ROOT / "data" / "dataset_info.json"
    ds_info = json.loads(ds_info_p.read_text(encoding="utf-8"))
    ds_info["gemma4_route2_stage1"] = {
        "file_name": "gemma4_route2_stage1/train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "tools": "tools", "audios": "audios"},
    }
    ds_info["gemma4_route2_stage1_valid"] = {
        "file_name": "gemma4_route2_stage1/valid.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "tools": "tools", "audios": "audios"},
    }
    ds_info_p.write_text(json.dumps(ds_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Print summary breakdown
    from collections import Counter
    cats = Counter(r["_meta"]["category"].split("_")[0] + "_" + r["_meta"]["category"].split("_")[1] for r in stage1_records)
    print("\n================ ROUTE 2 DATASET SUMMARY ================")
    print(f"Stage 1 Total: {len(clean_s1)} (Train: {len(s1_train)}, Valid: {len(s1_valid)}, Total 0.48s FLAC chunks: {audio_counter})")
    for k, v in sorted(cats.items()):
        print(f"  - {k}: {v}")
    u_lens = [len(r["speech_units"]) for r in stage2_records]
    print(f"Stage 2 Total: {len(stage2_records)} (Train: {len(s2_train)}, Valid: {len(s2_valid)})")
    print(f"  - Single Voice: {ASSISTANT_SINGLE_VOICE}")
    print(f"  - 25Hz Speech Units per utterance: min={min(u_lens)}, median={sorted(u_lens)[len(u_lens)//2]}, max={max(u_lens)}")
    print("=========================================================")


if __name__ == "__main__":
    main()
