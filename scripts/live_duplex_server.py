#!/usr/bin/env python3
"""
Gemma-4-E2B Route-1 v3.1 实时麦克风全双工语音对话 + 异步工具引擎 (macOS MPS / CUDA 通用)

已修复与增强：
1. 浏览器麦克风零回授音频图 (GainNode=0 隔离 destination) + 完整权限错误提示（若在内置 WebView 无麦克风权限会明确提示改用 Chrome/Safari）。
2. 自动近场增益校准 (AGC to RMS=0.09, 最大增益 4x)，解决笔记本麦克风原始音量偏小导致被模型误判为远场噪音 (<unused5>) 的问题。
3. 启动时自动预热 MPS/CUDA 音频与 LLM 着色器，消除首帧 3.8s 冷启动卡顿。
4. 异步非阻塞 TTS 队列 (带 epoch 代次号)，永不阻塞麦克风音频主循环；<unused6> 真打断 0ms 作废在途 TTS 并清空播放器。
5. 内置「🧪 一键模拟真实语音测试」与「实时文本/静音手动测试」，无需麦克风也能 1 秒验证全双工状态机 + 异步工具 + TTS 播报闭环！
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import edge_tts
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from peft import PeftModel
from starlette.websockets import WebSocketState
from transformers import AutoTokenizer, DynamicCache
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from llamafactory.data.tool_utils import Gemma4ToolUtils  # noqa: E402

SR = 16000
DUPLEX_TOKENS = [f"<unused{i}>" for i in range(8)]
SLOT_META = {
    "<unused0>": ("IDLE_SILENCE", "静音帧", "#64748b"),
    "<unused1>": ("USER_TALKING", "继续倾听 (防抢话)", "#38bdf8"),
    "<unused2>": ("START_SPEAK", "接话开口", "#22c55e"),
    "<unused3>": ("WAITING_TOOL", "后台工具运行中", "#f59e0b"),
    "<unused4>": ("IDLE_THINK", "答完空闲 (静音短路)", "#8b5cf6"),
    "<unused5>": ("IGNORE_NOISE", "抗噪/远场/回声忽略", "#06b6d4"),
    "<unused6>": ("INTERRUPTED", "近场真打断 (清播报)", "#ef4444"),
    "<unused7>": ("CANCEL_TOOL", "作废后台工具", "#f43f5e"),
}

LIVE_TOOLS_SCHEMA = [
    {"name": "get_weather", "description": "查询指定城市和日期的天气预报", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}, "required": ["city"]}},
    {"name": "set_alarm", "description": "设置闹钟或日程语音提醒", "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}, "required": ["time"]}},
    {"name": "control_smart_home", "description": "控制家中空调、灯光、窗帘或扫地机器人等智能设备", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "device": {"type": "string"}, "action": {"type": "string"}}, "required": ["device", "action"]}},
    {"name": "search_flight", "description": "查询两座城市之间的航班与票价信息", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination"]}},
    {"name": "navigate_to", "description": "规划驾车或公交导航路线并估算耗时", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "mode": {"type": "string"}}, "required": ["destination"]}},
    {"name": "play_music", "description": "按歌手、曲风或心情播放音乐", "parameters": {"type": "object", "properties": {"artist_or_mood": {"type": "string"}}, "required": ["artist_or_mood"]}},
    {"name": "track_package", "description": "查询快递物流最新配送进度", "parameters": {"type": "object", "properties": {"courier": {"type": "string"}}, "required": ["courier"]}},
    {"name": "translate_text", "description": "将口语内容翻译为指定目标语言", "parameters": {"type": "object", "properties": {"target_lang": {"type": "string"}, "phrase": {"type": "string"}}, "required": ["target_lang", "phrase"]}},
]


def patch_trainable_tokens_layer() -> None:
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer

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
                res = res + torch.matmul((x.unsqueeze(-1) == idx).to(res.dtype), deltas - base_W.index_select(0, idx).to(res.dtype))
            scale = self._get_embed_scale() if hasattr(self, "_get_embed_scale") else getattr(self.base_layer, "embed_scale", None)
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


def parse_gemma4_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    calls: List[Dict[str, Any]] = []
    pattern = re.compile(r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>", re.S)
    for m in pattern.finditer(text):
        fn_name = m.group(1)
        body = m.group(2)
        args: Dict[str, Any] = {}
        for km in re.finditer(r'([a-zA-Z0-9_]+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]+))', body):
            k = km.group(1)
            v = km.group(2) if km.group(2) is not None else km.group(3).strip()
            args[k] = v
        call_id = str(args.pop("id", f"c{len(calls)+1}"))
        calls.append({"id": call_id, "name": fn_name, "args": args})
    clean_text = pattern.sub("", text).strip()
    return clean_text, calls


async def execute_async_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(1.0)
    if name == "get_weather":
        return {"city": args.get("city", "北京"), "date": args.get("date", "今天"), "weather": "多云转晴", "temp": "19~26度", "aqi": 38}
    if name == "set_alarm":
        return {"ok": True, "time": args.get("time", "明早7点"), "label": args.get("label", "提醒")}
    if name == "control_smart_home":
        return {"ok": True, "room": args.get("room", "客厅"), "device": args.get("device", "空调"), "action": args.get("action", "打开")}
    if name == "search_flight":
        return {"origin": args.get("origin", "北京"), "destination": args.get("destination", "上海"), "flights": [{"no": "CA1501", "time": "09:30", "price": 760}]}
    if name == "navigate_to":
        return {"destination": args.get("destination", "首都机场"), "mode": args.get("mode", "开车"), "distance": "15.4公里", "duration": "26分钟", "traffic": "畅通"}
    if name == "play_music":
        return {"playing": True, "artist_or_mood": args.get("artist_or_mood", "轻音乐")}
    if name == "track_package":
        return {"courier": args.get("courier", "顺丰"), "status": "派送中", "eta": "今天下午三点前"}
    if name == "translate_text":
        return {"target_lang": args.get("target_lang", "英语"), "phrase": args.get("phrase", ""), "result": "Nice to meet you"}
    return {"ok": True, "tool": name, "args": args}


async def synth_edge_tts_mp3(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[bytes]:
    clean = re.sub(r"\[(breath|sigh|laughter)\]", "", text).strip()
    if not clean:
        return None
    try:
        comm = edge_tts.Communicate(clean, voice, rate="+6%")
        chunks: List[bytes] = []
        async for msg in comm.stream():
            if msg["type"] == "audio":
                chunks.append(msg["data"])
        return b"".join(chunks) if chunks else None
    except Exception:
        return None


class DuplexEngine:
    def __init__(self, base_model: str, adapter_repo: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        print(f"🚀 加载底座 {base_model} + 适配器 {adapter_repo} (设备: {self.device}, 精度: {self.dtype})...")

        patch_trainable_tokens_layer()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_repo)
        self.tokenizer.add_special_tokens({"additional_special_tokens": DUPLEX_TOKENS})
        self.fe = Gemma4AudioFeatureExtractor.from_pretrained(base_model)

        base = Gemma4ForConditionalGeneration.from_pretrained(base_model, dtype=self.dtype)
        peft_m = PeftModel.from_pretrained(base, adapter_repo)
        try:
            self.model = peft_m.merge_and_unload().to(self.device).eval()
            print("✅ LoRA + 8状态槽位已融合进原生底座 (merge_and_unload)！")
        except Exception as exc:
            print(f"⚠️ merge_and_unload 回退为 PeftModel ({exc})")
            self.model = peft_m.to(self.device).eval()

        self.turn_end_id = self.enc("<turn|>")[0]
        self.newline_id = self.enc("\n")[0]
        self.lock = threading.Lock()
        tool_decl = Gemma4ToolUtils.tool_formatter(LIVE_TOOLS_SCHEMA)
        self.sys_prompt_tools = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl}<turn|>\n"
        self.sys_prompt_chat = "<|turn>system\nYou are a real-time full-duplex voice assistant.<turn|>\n"
        self.enable_tools = False
        self.reset(enable_tools=False)
        # 启动时预热一次音频+LLM前向，消除首帧 MPS Metal JIT 卡顿！
        print("🔥 预热音频编码器与 KV Cache 着色器...")
        self.step_tick(np.random.randn(7680).astype(np.float32) * 0.05, silence_thresh=0.001)
        self.reset(enable_tools=False)
        print("✅ 预热完成！")

    def enc(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def reset(self, enable_tools: Optional[bool] = None) -> None:
        with self.lock:
            if enable_tools is not None:
                self.enable_tools = enable_tools
            sys_prompt = self.sys_prompt_tools if self.enable_tools else self.sys_prompt_chat
            self.cache = DynamicCache()
            sys_ids = [self.tokenizer.bos_token_id] + self.enc(sys_prompt)
            with torch.no_grad():
                inp = torch.tensor([sys_ids], dtype=torch.long, device=self.device)
                out = self.model(input_ids=inp, past_key_values=self.cache, use_cache=True)
                self.cache = out.past_key_values
            self.seq_len = len(sys_ids)
            self.is_idle = True
            self.prev_tail = np.zeros(int(SR * 0.16), dtype=np.float32)

    def step_tick(
        self,
        wav_in: Optional[np.ndarray],
        tool_obs: Optional[str] = None,
        text_in: Optional[str] = None,
        silence_thresh: float = 0.010,
        apply_agc: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            if self.seq_len > 3000:
                self.reset()

            rms_val = float(np.sqrt(np.mean(wav_in * wav_in) + 1e-12)) if wav_in is not None else 0.0
            is_silent = (wav_in is None) or (rms_val < silence_thresh)

            if tool_obs is None and text_in is None and is_silent and self.is_idle:
                if wav_in is not None:
                    self.prev_tail = wav_in[-len(self.prev_tail) :].copy()
                return {
                    "state": "<unused4>",
                    "text": "",
                    "raw": "<unused4>",
                    "tool_calls": [],
                    "cancel_ids": [],
                    "ms": 0.0,
                    "rms": round(rms_val, 4),
                    "seq_len": self.seq_len,
                    "short_circuit": True,
                }

            t0 = time.perf_counter()
            inp_feats, inp_mask = None, None
            if tool_obs is not None:
                chunk_ids = self.enc(f"<|turn>tool\n{tool_obs}<turn|>\n<|turn>model\n")
            elif text_in is not None:
                chunk_ids = self.enc(f"<|turn>user\n{text_in}<turn|>\n<|turn>model\n")
            elif is_silent:
                chunk_ids = self.enc("<|turn>user\n<unused0><turn|>\n<|turn>model\n")
                if wav_in is not None:
                    self.prev_tail = wav_in[-len(self.prev_tail) :].copy()
            else:
                wav_use = wav_in.astype(np.float32)
                # 笔记本麦克风近场自动增益校准（对齐训练集近场 RMS=0.09，上限 4.5x 避免把极小远场噪放大）
                if apply_agc and rms_val > 1e-5:
                    gain = min(4.5, 0.095 / rms_val)
                    wav_use = np.clip(wav_use * gain, -0.98, 0.98)
                wav_full = np.concatenate([self.prev_tail, wav_use]).astype(np.float32)
                self.prev_tail = wav_use[-len(self.prev_tail) :].copy()
                feat = self.fe([wav_full], sampling_rate=SR, return_tensors="pt")
                n_soft = count_audio_soft_tokens(feat["input_features_mask"][0])
                chunk_ids = (
                    self.enc("<|turn>user\n")
                    + [self.tokenizer.boa_token_id]
                    + [self.tokenizer.audio_token_id] * n_soft
                    + [self.tokenizer.eoa_token_id]
                    + self.enc("<turn|>\n<|turn>model\n")
                )
                inp_feats = feat["input_features"].to(self.device, self.dtype)
                inp_mask = feat["input_features_mask"].to(self.device)

            with torch.no_grad():
                inp = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
                pos = torch.arange(self.seq_len, self.seq_len + len(chunk_ids), dtype=torch.long, device=self.device).unsqueeze(0)
                attn = torch.ones((1, self.seq_len + len(chunk_ids)), dtype=torch.long, device=self.device)
                fwd_kw: Dict[str, Any] = {
                    "input_ids": inp,
                    "position_ids": pos,
                    "attention_mask": attn,
                    "past_key_values": self.cache,
                    "use_cache": True,
                }
                if inp_feats is not None:
                    fwd_kw["input_features"] = inp_feats
                    fwd_kw["input_features_mask"] = inp_mask
                out = self.model(**fwd_kw)
                self.cache = out.past_key_values
                self.seq_len += len(chunk_ids)

                gen_ids: List[int] = []
                next_id = int(out.logits[0, -1].argmax().item())
                for _ in range(52):
                    if next_id == self.turn_end_id or next_id == self.tokenizer.eos_token_id:
                        break
                    gen_ids.append(next_id)
                    step_inp = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    step_pos = torch.tensor([[self.seq_len]], dtype=torch.long, device=self.device)
                    step_attn = torch.ones((1, self.seq_len + 1), dtype=torch.long, device=self.device)
                    out = self.model(
                        input_ids=step_inp,
                        position_ids=step_pos,
                        attention_mask=step_attn,
                        past_key_values=self.cache,
                        use_cache=True,
                    )
                    self.cache = out.past_key_values
                    self.seq_len += 1
                    next_id = int(out.logits[0, -1].argmax().item())

                end_ids = [self.turn_end_id, self.newline_id]
                end_inp = torch.tensor([end_ids], dtype=torch.long, device=self.device)
                end_pos = torch.arange(self.seq_len, self.seq_len + 2, dtype=torch.long, device=self.device).unsqueeze(0)
                end_attn = torch.ones((1, self.seq_len + 2), dtype=torch.long, device=self.device)
                out = self.model(
                    input_ids=end_inp,
                    position_ids=end_pos,
                    attention_mask=end_attn,
                    past_key_values=self.cache,
                    use_cache=True,
                )
                self.cache = out.past_key_values
                self.seq_len += 2

            if self.device == "cuda":
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0

            first_tok = self.tokenizer.decode(gen_ids[:1], skip_special_tokens=False) if gen_ids else "<unused4>"
            raw_str = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            body = raw_str[len(first_tok) :] if first_tok in SLOT_META else raw_str

            cancel_ids: List[str] = []
            if "<unused7>" in raw_str:
                cancel_ids = re.findall(r"<unused7>([a-zA-Z0-9_]+)", raw_str)
                body = re.sub(r"<unused7>[a-zA-Z0-9_]+", "", body).strip()

            clean_text, tool_calls = parse_gemma4_tool_calls(body)
            self.is_idle = first_tok == "<unused4>"

            return {
                "state": first_tok if first_tok in SLOT_META else "<unused2>",
                "text": clean_text,
                "raw": raw_str,
                "tool_calls": tool_calls,
                "cancel_ids": cancel_ids,
                "ms": round(ms, 1),
                "rms": round(rms_val, 4),
                "seq_len": self.seq_len,
                "short_circuit": False,
            }


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Gemma-4-E2B Route-1 全双工实时语音助手</title>
<style>
  :root { --bg:#0b1120; --card:#111827; --border:#1f2937; --text:#f3f4f6; --muted:#9ca3af; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width:980px; margin:0 auto; padding:22px 18px 48px; }
  h1 { font-size:21px; margin:0 0 6px; display:flex; align-items:center; gap:10px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
  #errBanner { display:none; background:#450a0a; border:1px solid #ef4444; color:#fecaca; padding:12px 14px; border-radius:10px; font-size:13px; margin-bottom:14px; line-height:1.5; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:var(--card); border:1px solid var(--border); padding:14px 16px; border-radius:14px; margin-bottom:14px; }
  button { border:none; border-radius:10px; padding:10px 16px; font-size:13px; font-weight:600; cursor:pointer; transition:.15s; }
  #btnMic { background:#22c55e; color:#052e16; }
  #btnMic.on { background:#ef4444; color:#fff; }
  #btnDemo { background:#0284c7; color:#fff; }
  #btnReset { background:#334155; color:#f8fafc; }
  label { font-size:13px; color:var(--muted); display:flex; align-items:center; gap:6px; }
  .slots { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:14px; }
  .slot { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:8px 10px; font-size:12px; opacity:.45; transition:.12s; }
  .slot.active { opacity:1; transform:translateY(-1px); box-shadow:0 0 0 1px currentColor inset; }
  .slot b { display:block; font-size:13px; margin-bottom:2px; }
  .metrics { display:flex; flex-wrap:wrap; gap:16px; font-size:13px; color:var(--muted); margin-bottom:12px; }
  .metrics span b { color:#f8fafc; }
  .meter-bg { height:8px; background:#1e293b; border-radius:99px; overflow:hidden; margin-bottom:14px; }
  #meterBar { height:100%; width:0%; background:linear-gradient(90deg,#38bdf8,#22c55e,#f59e0b); transition:width .08s; }
  .grid { display:grid; grid-template-columns:2fr 1fr; gap:14px; }
  .panel { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; min-height:320px; max-height:460px; overflow-y:auto; }
  .panel h3 { margin:0 0 10px; font-size:14px; color:#cbd5e1; }
  .log-line { font-size:13px; line-height:1.55; padding:6px 8px; border-radius:8px; margin-bottom:6px; background:#0f172a; }
  .tool-item { font-size:12px; padding:8px; border-radius:8px; margin-bottom:8px; background:#0f172a; border-left:3px solid #f59e0b; }
  .tool-item.done { border-left-color:#22c55e; }
  .tool-item.cancelled { border-left-color:#ef4444; opacity:.75; text-decoration:line-through; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎙️ Gemma-4-E2B Route-1 全双工原生语音助手 <span style="font-size:12px;background:#1e293b;padding:3px 8px;border-radius:99px;color:#38bdf8;">v3.1 · 零 ASR · 增量 KV Cache</span></h1>
  <div class="sub">每 480ms 原始声波直入冻结的 300M gemma4_audio 编码器 · 支持随时真打断、自回声/远场抗噪、非阻塞异步工具与中途作废</div>

  <div id="errBanner"></div>

  <div class="controls">
    <button id="btnMic">🎤 开始麦克风全双工</button>
    <button id="btnDemo">🧪 一键真实语音闭环自测</button>
    <button id="btnReset">🔄 重置上下文</button>
    <label>模式 <select id="selMode" style="background:#1e293b;color:#f8fafc;border:1px solid #334155;border-radius:8px;padding:6px 8px;">
      <option value="chat" selected>💬 纯语音闲聊 (日常对话/问答)</option>
      <option value="tools">🛠️ 语音工具模式 (天气/空调/音乐/航班)</option>
    </select></label>
    <label><input type="checkbox" id="chkAec" checked /> 浏览器 AEC</label>
    <label>静音阈值 <input type="range" id="rngThresh" min="0.003" max="0.04" step="0.001" value="0.010" /><span id="lblThresh">0.010</span></label>
  </div>

  <div class="meter-bg"><div id="meterBar"></div></div>
  <div class="metrics">
    <span>连接: <b id="stConn">未连接</b></span>
    <span>麦克风: <b id="stMic">未开启</b></span>
    <span>单帧延迟: <b id="stMs">-</b></span>
    <span>上下文: <b id="stSeq">0</b> tok</span>
    <span>RMS: <b id="stRms">0.0000</b></span>
  </div>

  <div class="slots" id="slotGrid">
    <div class="slot" data-tok="&lt;unused0&gt;" style="color:#94a3b8"><b>&lt;unused0&gt;</b>静音帧输入</div>
    <div class="slot" data-tok="&lt;unused1&gt;" style="color:#38bdf8"><b>&lt;unused1&gt;</b>继续倾听 (防抢话)</div>
    <div class="slot" data-tok="&lt;unused2&gt;" style="color:#22c55e"><b>&lt;unused2&gt;</b>接话开口播报</div>
    <div class="slot" data-tok="&lt;unused3&gt;" style="color:#f59e0b"><b>&lt;unused3&gt;</b>异步工具等待中</div>
    <div class="slot" data-tok="&lt;unused4&gt;" style="color:#a78bfa"><b>&lt;unused4&gt;</b>空闲 (0ms短路)</div>
    <div class="slot" data-tok="&lt;unused5&gt;" style="color:#06b6d4"><b>&lt;unused5&gt;</b>抗噪/回声忽略</div>
    <div class="slot" data-tok="&lt;unused6&gt;" style="color:#ef4444"><b>&lt;unused6&gt;</b>真打断 (清播报)</div>
    <div class="slot" data-tok="&lt;unused7&gt;" style="color:#f43f5e"><b>&lt;unused7&gt;</b>作废后台工具</div>
  </div>

  <div class="grid">
    <div class="panel"><h3>💬 实时状态流与助手回复</h3><div id="chatLog"></div></div>
    <div class="panel"><h3>🛠️ 异步后台工具队列</h3><div id="toolLog"></div></div>
  </div>
</div>

<script>
const SR = 16000, CHUNK_SAMPLES = 7680;
let ws = null, audioCtx = null, micStream = null, processor = null, zeroGain = null;
let pcmBuf = new Float32Array(0);
let ttsCtx = null, activeSources = [], nextPlayTime = 0, micOn = false;

const btnMic = document.getElementById("btnMic");
const btnDemo = document.getElementById("btnDemo");
const btnReset = document.getElementById("btnReset");
const errBanner = document.getElementById("errBanner");
const selMode = document.getElementById("selMode");
const rngThresh = document.getElementById("rngThresh");
const lblThresh = document.getElementById("lblThresh");

function esc(s) { return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
selMode.onchange = () => {
  flushTTS();
  document.getElementById("chatLog").innerHTML = "";
  document.getElementById("toolLog").innerHTML = "";
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:"reset", enable_tools: selMode.value === "tools"}));
};

function showErr(msg) {
  errBanner.style.display = msg ? "block" : "none";
  errBanner.innerHTML = msg || "";
}
rngThresh.oninput = () => {
  lblThresh.textContent = rngThresh.value;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:"config", silence_thresh: parseFloat(rngThresh.value)}));
};
function highlightSlot(tok) {
  document.querySelectorAll(".slot").forEach(el => el.classList.toggle("active", el.dataset.tok === tok));
}
function addLog(html) {
  const box = document.getElementById("chatLog");
  const d = document.createElement("div");
  d.className = "log-line";
  d.innerHTML = html;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}
function upsertTool(id, html, cls) {
  const box = document.getElementById("toolLog");
  let el = document.getElementById("tool-" + id);
  if (!el) { el = document.createElement("div"); el.id = "tool-" + id; box.prepend(el); }
  el.className = "tool-item " + (cls || "");
  el.innerHTML = html;
}
function flushTTS() {
  for (const s of activeSources) { try { s.stop(); } catch {} }
  activeSources = [];
  if (ttsCtx) nextPlayTime = ttsCtx.currentTime;
}
async function playMp3B64(b64) {
  try {
    if (!ttsCtx) ttsCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (ttsCtx.state === "suspended") await ttsCtx.resume();
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const buf = await ttsCtx.decodeAudioData(bytes.buffer.slice(0));
    const src = ttsCtx.createBufferSource();
    src.buffer = buf;
    src.connect(ttsCtx.destination);
    const startAt = Math.max(ttsCtx.currentTime + 0.02, nextPlayTime);
    src.start(startAt);
    nextPlayTime = startAt + buf.duration;
    activeSources.push(src);
    src.onended = () => { activeSources = activeSources.filter(x => x !== src); };
  } catch (e) {
    console.error("TTS playback error:", e);
  }
}

function ensureWS() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === 1) return resolve(ws);
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => {
      document.getElementById("stConn").textContent = "已连接";
      ws.send(JSON.stringify({type:"config", silence_thresh: parseFloat(rngThresh.value)}));
      ws.send(JSON.stringify({type:"reset", enable_tools: selMode.value === "tools"}));
      resolve(ws);
    };
    ws.onerror = (e) => { showErr("WebSocket 连接失败，请确认本地服务正在运行。"); reject(e); };
    ws.onclose = () => { document.getElementById("stConn").textContent = "已断开"; };
    ws.onmessage = async (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "info") {
        addLog(`<span style="color:#38bdf8">[系统]</span> ${esc(msg.text)}`);
      } else if (msg.type === "tick") {
        highlightSlot(msg.state);
        document.getElementById("stMs").textContent = msg.short_circuit ? "0 ms (静音短路)" : `${msg.ms} ms`;
        if (msg.seq_len) document.getElementById("stSeq").textContent = msg.seq_len;
        document.getElementById("stRms").textContent = (msg.rms || 0).toFixed(4);
        document.getElementById("meterBar").style.width = Math.min(100, (msg.rms || 0) * 900) + "%";
        if (msg.text) {
          addLog(`<span style="color:#22c55e">[${esc(msg.state)}]</span> ${esc(msg.text)} <span style="color:#64748b;font-size:11px">(${msg.ms}ms)</span>`);
        } else if (msg.state === "<unused6>") {
          addLog(`<span style="color:#ef4444">[真打断 &lt;unused6&gt;]</span> 已立即闭嘴并清空播报队列！`);
        } else if (msg.state === "<unused5>") {
          addLog(`<span style="color:#06b6d4">[抗噪忽略 &lt;unused5&gt;]</span> 检测到远场旁人/杂音/回声，继续保持不断流`);
        } else if (msg.state === "<unused1>" && !msg.short_circuit) {
          addLog(`<span style="color:#38bdf8">[倾听 &lt;unused1&gt;]</span> 正在听你说话... <span style="color:#64748b;font-size:11px">(RMS=${(msg.rms||0).toFixed(3)}, ${msg.ms}ms)</span>`);
        } else if (msg.state === "<unused3>") {
          addLog(`<span style="color:#f59e0b">[等待工具 &lt;unused3&gt;]</span> 后台工具异步执行中...`);
        }
      } else if (msg.type === "flush_tts") {
        flushTTS();
      } else if (msg.type === "tts") {
        await playMp3B64(msg.mp3_b64);
      } else if (msg.type === "tool_start") {
        upsertTool(msg.id, `⏳ <b>${msg.name}</b> (${msg.id})<br/>参数: ${JSON.stringify(msg.args)}`, "");
      } else if (msg.type === "tool_done") {
        upsertTool(msg.id, `✅ <b>${msg.name}</b> (${msg.id})<br/>结果: ${JSON.stringify(msg.result)}`, "done");
      } else if (msg.type === "tool_cancel") {
        upsertTool(msg.id, `🚫 <b>${msg.id}</b> 已被用户打断实时作废 (&lt;unused7&gt;)`, "cancelled");
      }
    };
  });
}

function resampleLinear(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio, idx = Math.floor(pos), frac = pos - idx;
    out[i] = input[idx] * (1 - frac) + (input[idx + 1] || input[idx]) * frac;
  }
  return out;
}

async function startMic() {
  showErr("");
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("当前浏览器环境不支持麦克风 API（如果你在 IDE 内置预览页打开，请复制 http://127.0.0.1:8765 到系统 Chrome 或 Safari 中打开！）");
    }
    await ensureWS();
    if (!ttsCtx) ttsCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (ttsCtx.state === "suspended") await ttsCtx.resume();

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: document.getElementById("chkAec").checked,
        noiseSuppression: false,
        autoGainControl: true,
        channelCount: 1,
      }
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    const src = audioCtx.createMediaStreamSource(micStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    zeroGain = audioCtx.createGain();
    zeroGain.gain.value = 0; // 关键修复：彻底隔离麦克风回授到扬声器！
    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== 1 || !micOn) return;
      const raw = e.inputBuffer.getChannelData(0);
      const r16 = resampleLinear(raw, audioCtx.sampleRate, SR);
      const merged = new Float32Array(pcmBuf.length + r16.length);
      merged.set(pcmBuf);
      merged.set(r16, pcmBuf.length);
      let offset = 0;
      while (merged.length - offset >= CHUNK_SAMPLES) {
        const slice = merged.slice(offset, offset + CHUNK_SAMPLES);
        ws.send(slice.buffer);
        offset += CHUNK_SAMPLES;
      }
      pcmBuf = merged.slice(offset);
    };
    src.connect(processor);
    processor.connect(zeroGain);
    zeroGain.connect(audioCtx.destination);

    micOn = true;
    document.getElementById("stMic").textContent = "录音采集中";
    btnMic.textContent = "⏹️ 停止麦克风";
    btnMic.classList.add("on");
    addLog(`<span style="color:#22c55e">[麦克风已开启]</span> 请直接开口说话（例如：“帮我查一下明天杭州的天气”），说完停顿约 0.5 秒模型即刻接话！`);
  } catch (err) {
    showErr(`❌ 麦克风启动失败：<b>${err.message || err}</b><br/>提示：如果在内置 WebView 预览里，请改用 <b>Chrome / Safari</b> 打开 <code>http://127.0.0.1:8765</code>，或者先点蓝色按钮 <b>「🧪 一键真实语音闭环自测」</b> 体验模型全双工+语音播报！`);
    stopMic();
  }
}

function stopMic() {
  micOn = false;
  flushTTS();
  if (processor) { processor.disconnect(); processor = null; }
  if (zeroGain) { zeroGain.disconnect(); zeroGain = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  pcmBuf = new Float32Array(0);
  document.getElementById("stMic").textContent = "未开启";
  btnMic.textContent = "🎤 开始麦克风全双工";
  btnMic.classList.remove("on");
}

btnMic.onclick = () => (micOn ? stopMic() : startMic());

btnDemo.onclick = async () => {
  showErr("");
  try {
    await ensureWS();
    if (!ttsCtx) ttsCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (ttsCtx.state === "suspended") await ttsCtx.resume();
    ws.send(JSON.stringify({type:"demo_replay"}));
  } catch (e) {
    showErr("无法触发闭环自测: " + e);
  }
};

btnReset.onclick = async () => {
  flushTTS();
  document.getElementById("chatLog").innerHTML = "";
  document.getElementById("toolLog").innerHTML = "";
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({type:"reset", enable_tools: selMode.value === "tools"}));
};
</script>
</body>
</html>
"""


def create_app(engine: DuplexEngine) -> FastAPI:
    app = FastAPI(title="Gemma-4-E2B Route-1 Live Full-Duplex")

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        silence_thresh = 0.010
        pending_tool_results: asyncio.Queue[str] = asyncio.Queue()
        running_tools: Dict[str, asyncio.Task] = {}
        tts_buffer: List[str] = []
        tts_epoch = 0
        tts_tasks: List[asyncio.Task] = []

        async def safe_send(payload: Dict[str, Any]) -> bool:
            if ws.client_state != WebSocketState.CONNECTED:
                return False
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
                return True
            except Exception:
                return False

        def spawn_tts(text_to_speak: str) -> None:
            my_epoch = tts_epoch

            async def _bg() -> None:
                mp3 = await synth_edge_tts_mp3(text_to_speak)
                if mp3 and my_epoch == tts_epoch:
                    await safe_send({"type": "tts", "mp3_b64": base64.b64encode(mp3).decode()})

            tts_tasks.append(asyncio.create_task(_bg()))

        def flush_tts_buffer(force: bool = False) -> None:
            joined = "".join(tts_buffer).strip()
            if not joined:
                return
            if not force and len(joined) < 8 and not any(joined.endswith(p) for p in ("。", "！", "？", "，", "；")):
                return
            tts_buffer.clear()
            spawn_tts(joined)

        async def run_tool_bg(cid: str, name: str, args: Dict[str, Any]) -> None:
            try:
                await safe_send({"type": "tool_start", "id": cid, "name": name, "args": args})
                res = await execute_async_tool(name, args)
                await safe_send({"type": "tool_done", "id": cid, "name": name, "result": res})
                await pending_tool_results.put(json.dumps(res, ensure_ascii=False))
            except asyncio.CancelledError:
                await safe_send({"type": "tool_cancel", "id": cid})
                raise
            finally:
                running_tools.pop(cid, None)

        async def handle_step_result(res: Dict[str, Any]) -> None:
            nonlocal tts_epoch
            await safe_send({"type": "tick", **res})
            if res["state"] == "<unused6>":
                tts_epoch += 1
                tts_buffer.clear()
                await safe_send({"type": "flush_tts"})
                for cid in res["cancel_ids"]:
                    if cid in running_tools:
                        running_tools[cid].cancel()
                return
            for tc in res["tool_calls"]:
                cid = tc["id"]
                running_tools[cid] = asyncio.create_task(run_tool_bg(cid, tc["name"], tc["args"]))
            if res["text"]:
                tts_buffer.append(res["text"])
                flush_tts_buffer(force=bool(res["tool_calls"]))
            elif res["state"] == "<unused4>":
                flush_tts_buffer(force=True)

        audio_q: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue()

        async def receiver_loop() -> None:
            nonlocal silence_thresh, tts_epoch
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        await audio_q.put(None)
                        return
                    if msg.get("text"):
                        data = json.loads(msg["text"])
                        t = data.get("type")
                        if t == "config":
                            silence_thresh = float(data.get("silence_thresh", 0.010))
                        elif t == "reset":
                            tts_epoch += 1
                            for tk in list(running_tools.values()):
                                tk.cancel()
                            running_tools.clear()
                            tts_buffer.clear()
                            await audio_q.put(("reset", data.get("enable_tools")))
                        elif t == "demo_replay":
                            await audio_q.put(("demo", None))
                        continue
                    raw_bytes = msg.get("bytes")
                    if raw_bytes:
                        await audio_q.put(("audio", np.frombuffer(raw_bytes, dtype=np.float32).copy()))
            except Exception:
                await audio_q.put(None)

        recv_task = asyncio.create_task(receiver_loop())
        try:
            while True:
                item = await audio_q.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "reset":
                    await asyncio.to_thread(engine.reset, payload)
                    mode_lbl = "🛠️ 语音工具模式" if engine.enable_tools else "💬 纯语音闲聊模式"
                    await safe_send({"type": "info", "text": f"上下文已重置（当前：{mode_lbl}）。"})
                    continue
                if kind == "demo":
                    await asyncio.to_thread(engine.reset, True)
                    ood_p = REPO_ROOT / "data/gemma4_duplex_route1_v3p1/ood_eval.json"
                    if not ood_p.exists():
                        await safe_send({"type": "info", "text": "未找到本地 OOD 演示音频清单。"})
                        continue
                    manifest = json.loads(ood_p.read_text(encoding="utf-8"))
                    case = manifest["cross_domain_tool_cases"][2]  # 杭州后天会下雨吗帮我看看
                    await safe_send({"type": "info", "text": f"🧪 开始零 ASR 真实语音闭环自测（未见音色原句：「{case['transcript']}」）"})
                    for rel in case["part_paths"]:
                        w, _ = sf.read(str(REPO_ROOT / "data" / rel), dtype="float32")
                        r = await asyncio.to_thread(engine.step_tick, w, None, None, 0.005, False)
                        await handle_step_result(r)
                    # 说完静音 -> 触发垫话 + 异步工具
                    r_go = await asyncio.to_thread(engine.step_tick, None, None, None, silence_thresh, False)
                    await handle_step_result(r_go)
                    # 等待后台工具执行，并连续推进直到答完 <unused4>
                    for _ in range(10):
                        await asyncio.sleep(0.45)
                        obs = await pending_tool_results.get() if not pending_tool_results.empty() else None
                        r_n = await asyncio.to_thread(engine.step_tick, None, obs, None, silence_thresh, False)
                        await handle_step_result(r_n)
                        if r_n["state"] == "<unused4>":
                            break
                    continue

                # 音频帧实时抗堆积
                batch_wavs = [payload]
                while not audio_q.empty():
                    nxt = audio_q.get_nowait()
                    if nxt is None:
                        return
                    if nxt[0] == "audio":
                        batch_wavs.append(nxt[1])
                    else:
                        await audio_q.put(nxt)
                        break

                speech_chunks = [w for w in batch_wavs if float(np.sqrt(np.mean(w * w) + 1e-12)) >= silence_thresh]
                wav_in = np.concatenate(speech_chunks[-3:]).astype(np.float32) if speech_chunks else batch_wavs[-1]

                tool_obs = await pending_tool_results.get() if not pending_tool_results.empty() else None
                res = await asyncio.to_thread(engine.step_tick, wav_in, tool_obs, None, silence_thresh, True)
                await handle_step_result(res)
        except WebSocketDisconnect:
            pass
        finally:
            recv_task.cancel()
            for tk in list(running_tools.values()) + tts_tasks:
                tk.cancel()

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, default="google/gemma-4-E2B-it")
    p.add_argument("--adapter", type=str, default="kongDeQiang/gemma4-e2b-duplex-route1-v3p1-lora")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    engine = DuplexEngine(args.base_model, args.adapter)
    app = create_app(engine)
    print(f"\n🎙️ 实时全双工语音界面已就绪！请在 Chrome/Safari 打开: http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
