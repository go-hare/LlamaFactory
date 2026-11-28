#!/usr/bin/env python3
"""
Gemma-4-E2B Route 2 (II-A) End-to-End Three-Stage Streaming Speech-to-Speech Server (CUDA <200ms TTFA):
  [16kHz 0.48s Mic PCM]
      -> Stage 1: Gemma-4-E2B Full-Duplex Thinker (8 state slots <unused0~7> + zero-ASR async <tool_call>)
      -> Stage 2: Gemma4SpeechGenerator Talker (1536-dim continuous hidden_states + Gate-Fusion -> 25Hz units)
      -> Stage 3: CosyVoice2 GPU Flow-HiFT Streaming Vocoder (-> gapless 24kHz PCM WebAudio stream + 0ms barge-in!)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

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

from gemma4_omni_route2 import (  # noqa: E402
    Cosy2StreamingVocoder,
    Gemma4SpeechGenerator,
    TalkerStreamSession,
    ensure_gemma4_duplex_special_tokens,
)
from llamafactory.data.tool_utils import Gemma4ToolUtils  # noqa: E402


SR_IN = 16000
SR_OUT = 24000
CHUNK_SAMPLES = 7680  # 0.48s @ 16kHz

LIVE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "查询指定城市和日期的天气预报",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "set_alarm",
        "description": "设置闹钟或日程语音提醒",
        "parameters": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}, "required": ["time"]},
    },
    {
        "name": "control_smart_home",
        "description": "控制家中空调、灯光、窗帘或扫地机器人等智能设备",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "device": {"type": "string"}, "action": {"type": "string"}}, "required": ["device", "action"]},
    },
    {
        "name": "search_flight",
        "description": "查询两座城市之间的航班与票价信息",
        "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination"]},
    },
    {
        "name": "navigate_to",
        "description": "规划驾车或公交导航路线并估算耗时",
        "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "mode": {"type": "string"}}, "required": ["destination"]},
    },
    {
        "name": "play_music",
        "description": "按歌手、曲风或心情播放音乐",
        "parameters": {"type": "object", "properties": {"artist_or_mood": {"type": "string"}}, "required": ["artist_or_mood"]},
    },
    {
        "name": "track_package",
        "description": "查询快递物流最新配送进度",
        "parameters": {"type": "object", "properties": {"courier": {"type": "string"}}, "required": ["courier"]},
    },
    {
        "name": "translate_text",
        "description": "将口语内容翻译为指定目标语言",
        "parameters": {"type": "object", "properties": {"target_lang": {"type": "string"}, "phrase": {"type": "string"}}, "required": ["target_lang", "phrase"]},
    },
    {
        "name": "check_account_balance",
        "description": "查询银行卡余额、账单明细或卡片状态",
        "parameters": {"type": "object", "properties": {"query_type": {"type": "string"}}, "required": ["query_type"]},
    },
]


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
        calls.append({"id": f"c{len(calls)+1}", "name": fn_name, "args": args})
    clean = pattern.sub("", text)
    clean = re.sub(r"<unused\d+>", "", clean).strip()
    return clean, calls


async def execute_async_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.8)
    if name == "get_weather":
        return {"city": args.get("city", "北京"), "date": args.get("date", "明天"), "weather": "多云转晴", "temp": "19到26度", "aqi": 36}
    if name == "set_alarm":
        return {"ok": True, "time": args.get("time", "明早七点"), "label": args.get("label", "提醒")}
    if name == "control_smart_home":
        return {"ok": True, "room": args.get("room", "客厅"), "device": args.get("device", "空调"), "action": args.get("action", "打开")}
    if name == "search_flight":
        return {"origin": args.get("origin", "北京"), "destination": args.get("destination", "三亚"), "flights": [{"no": "CA1502", "time": "09:30", "price": 760}]}
    if name == "navigate_to":
        return {"destination": args.get("destination", "首都国际机场"), "mode": args.get("mode", "开车"), "duration": "26分钟", "traffic": "畅通"}
    if name == "play_music":
        return {"playing": True, "artist_or_mood": args.get("artist_or_mood", "轻音乐")}
    if name == "track_package":
        return {"courier": args.get("courier", "顺丰"), "status": "派送中", "eta": "今天下午四点前"}
    if name == "translate_text":
        return {"target_lang": args.get("target_lang", "英语"), "result": "Excuse me, where is the subway station?"}
    if name == "check_account_balance":
        return {"status": "正常", "overdue": False, "balance_ok": True}
    return {"ok": True, "tool": name, "args": args}


class Route2OmniEngine:
    """Thread-safe GPU engine uniting Stage-1 Gemma-4 Thinker + Stage-2 Talker + Stage-3 CosyVoice2 Vocoder."""

    def __init__(
        self,
        base_model: str,
        stage1_adapter: str,
        stage2_talker: str,
        prompt_wav: Optional[str] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        print(f"🚀 [1/3] Loading Stage-1 Thinker ({base_model} + {stage1_adapter}) on {self.device}...")

        patch_trainable_tokens_layer()
        tok_src = stage1_adapter if stage1_adapter else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        ensure_gemma4_duplex_special_tokens(self.tokenizer)
        self.fe = Gemma4AudioFeatureExtractor.from_pretrained(base_model)

        thinker = Gemma4ForConditionalGeneration.from_pretrained(base_model, dtype=self.dtype)
        if stage1_adapter:
            peft_m = PeftModel.from_pretrained(thinker, stage1_adapter)
            try:
                thinker = peft_m.merge_and_unload()
                print("   ✅ Merged Stage-1 LoRA + 8 state tokens into native Gemma-4 weights!")
            except Exception as exc:
                print(f"   ⚠️ Fallback to PeftModel ({exc})")
                thinker = peft_m
        self.thinker = thinker.to(self.device).eval()

        print(f"🚀 [2/3] Loading Stage-2 Talker ({stage2_talker or 'LLaMA-Omni2-1.5B-Bilingual'})...")
        if stage2_talker:
            self.talker = Gemma4SpeechGenerator.from_checkpoint(stage2_talker, torch_dtype=self.dtype, device=self.device).eval()
        else:
            self.talker = Gemma4SpeechGenerator.from_llama_omni2_pretrained(torch_dtype=self.dtype).to(self.device).eval()

        voc_dev = "cuda" if self.device == "cuda" else "cpu"
        print(f"🚀 [3/3] Loading Stage-3 CosyVoice2 24kHz Streaming Vocoder on {voc_dev}...")
        # Use official LLaMA-Omni2 zero-prompt x-vector streaming mode (5.6x faster first-packet ODE!)
        self.vocoder = Cosy2StreamingVocoder(device=voc_dev, prompt_wav_path=prompt_wav, use_zero_shot_prompt=False)
        self.voc_cuda_stream = torch.cuda.Stream() if self.device == "cuda" else None

        self.turn_end_id = self.enc("<turn|>")[0]
        self.newline_id = self.enc("\n")[0]
        self.tc_end_id = self.tokenizer.convert_tokens_to_ids("<tool_call|>")
        tool_decl = Gemma4ToolUtils.tool_formatter(LIVE_TOOLS_SCHEMA)
        self.sys_prompt = f"<|turn>system\nYou are a real-time full-duplex voice assistant.{tool_decl}<turn|>\n"
        self.lock = threading.Lock()
        self.reset_context()
        self._warmup()

    def enc(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def _crop_cache(self, target_len: int) -> None:
        if hasattr(self.cache, "crop") and hasattr(self.cache, "get_seq_length"):
            remove_n = int(self.cache.get_seq_length()) - int(target_len)
            if remove_n > 0:
                self.cache.crop(-remove_n)
        self.seq_len = target_len

    def _warmup(self) -> None:
        print("🔥 Warming up Thinker + Talker + Vocoder CUDA kernels...")
        dummy_wav = (np.random.randn(CHUNK_SAMPLES).astype(np.float32) * 0.04)
        _ = self.prefill_audio_tick(dummy_wav)
        ts = TalkerStreamSession()
        vs = self.vocoder.init_session()
        u = self.talker.step_gemma_token(self.tokenizer, self.enc("你好")[0], torch.zeros(1536, device=self.device, dtype=self.dtype), ts, is_turn_end=False)
        u2 = self.talker.step_gemma_token(self.tokenizer, self.turn_end_id, torch.zeros(1536, device=self.device, dtype=self.dtype), ts, is_turn_end=True)
        _ = self.vocoder.push_units(u + u2, vs, finalize=True)
        self.reset_context()
        print("✅ All 3 stages warmed up and ready!")

    def reset_context(self) -> None:
        with self.lock:
            self.cache = DynamicCache()
            sys_ids = [self.tokenizer.bos_token_id] + self.enc(self.sys_prompt)
            with torch.inference_mode():
                inp = torch.tensor([sys_ids], dtype=torch.long, device=self.device)
                out = self.thinker(input_ids=inp, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                self.cache = out.past_key_values
            self.seq_len = len(sys_ids)

    @torch.inference_mode()
    def prefill_audio_tick(self, wav_048s: np.ndarray, rollback_on_ignore: bool = False) -> Tuple[str, float]:
        """Fast (~11ms on A100) single-chunk audio state check (<unused1> listen, <unused5> ignore, <unused6> barge-in)."""
        with self.lock:
            if self.seq_len > 2800:
                self.reset_context()
            prev_seq_len = self.seq_len
            t0 = time.perf_counter()
            rms = float(np.sqrt(np.mean(wav_048s**2) + 1e-12))
            wav_use = wav_048s.astype(np.float32)
            if rms >= 0.018:
                gain = min(3.2, 0.085 / rms)
                wav_use = np.clip(wav_use * gain, -0.96, 0.96)

            feat = self.fe([wav_use], sampling_rate=SR_IN, return_tensors="pt")
            n_soft = count_audio_soft_tokens(feat["input_features_mask"][0])
            chunk_ids = (
                self.enc("<|turn>user\n")
                + [self.tokenizer.boa_token_id]
                + [self.tokenizer.audio_token_id] * n_soft
                + [self.tokenizer.eoa_token_id]
                + self.enc("<turn|>\n<|turn>model\n")
            )
            inp = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
            pos = torch.arange(self.seq_len, self.seq_len + len(chunk_ids), dtype=torch.long, device=self.device).unsqueeze(0)
            attn = torch.ones((1, self.seq_len + len(chunk_ids)), dtype=torch.long, device=self.device)
            out = self.thinker(
                input_ids=inp,
                position_ids=pos,
                attention_mask=attn,
                input_features=feat["input_features"].to(self.device, self.dtype),
                input_features_mask=feat["input_features_mask"].to(self.device),
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )
            self.cache = out.past_key_values
            self.seq_len += len(chunk_ids)

            first_id = int(out.logits[0, -1].argmax().item())
            # Roll back ambient noise / echo frames (<unused5>) so idle KV cache stays 100% pure!
            if rollback_on_ignore and first_id not in (7, 12):  # 7=<unused1>, 12=<unused6>
                self._crop_cache(prev_seq_len)
                raw_state = self.tokenizer.decode([first_id], skip_special_tokens=False)
                return raw_state, (time.perf_counter() - t0) * 1000.0

            gen_ids = [first_id]
            # If <unused6> barge-in, check next 1..3 tokens in case of <unused6><unused1> or <unused6><unused7>c1
            cur_id = first_id
            if first_id == 12:  # <unused6>
                for _ in range(3):
                    s_inp = torch.tensor([[cur_id]], dtype=torch.long, device=self.device)
                    s_pos = torch.tensor([[self.seq_len]], dtype=torch.long, device=self.device)
                    s_attn = torch.ones((1, self.seq_len + 1), dtype=torch.long, device=self.device)
                    s_out = self.thinker(
                        input_ids=s_inp, position_ids=s_pos, attention_mask=s_attn,
                        past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                    )
                    self.cache = s_out.past_key_values
                    self.seq_len += 1
                    cur_id = int(s_out.logits[0, -1].argmax().item())
                    if cur_id in (self.turn_end_id, self.tokenizer.eos_token_id):
                        break
                    gen_ids.append(cur_id)
                tail = [self.turn_end_id, self.newline_id]
            else:
                tail = [first_id, self.turn_end_id, self.newline_id]

            t_inp = torch.tensor([tail], dtype=torch.long, device=self.device)
            t_pos = torch.arange(self.seq_len, self.seq_len + len(tail), dtype=torch.long, device=self.device).unsqueeze(0)
            t_attn = torch.ones((1, self.seq_len + len(tail)), dtype=torch.long, device=self.device)
            t_out = self.thinker(
                input_ids=t_inp, position_ids=t_pos, attention_mask=t_attn,
                past_key_values=self.cache, use_cache=True, logits_to_keep=1,
            )
            self.cache = t_out.past_key_values
            self.seq_len += len(tail)

            raw_state = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            return raw_state, (time.perf_counter() - t0) * 1000.0

    def stream_response_generator(
        self,
        trigger_mode: str = "silence",
        tool_obs: Optional[str] = None,
        text_query: Optional[str] = None,
        abort_flag: Optional[threading.Event] = None,
    ):
        """Dual-thread pipelined streaming: Thinker+Talker producer || Vocoder CUDA-stream consumer (zero buffer underrun)."""
        t_start = time.perf_counter()
        event_q: queue.Queue = queue.Queue()
        unit_q: queue.Queue = queue.Queue()
        voc_done_evt = threading.Event()
        first_audio_box: List[Optional[float]] = [None]
        voc_sess = self.vocoder.init_session()

        def _vocoder_worker() -> None:
            try:
                ctx = torch.cuda.stream(self.voc_cuda_stream) if self.voc_cuda_stream is not None else torch.inference_mode()
                with ctx, torch.inference_mode():
                    while True:
                        item = unit_q.get()
                        if item is None:
                            self.vocoder.abort_session(voc_sess)
                            break
                        batch_u, saw_end = list(item[0]), bool(item[1])
                        # Coalesce all units already generated while previous chunk was synthesizing!
                        while not saw_end:
                            try:
                                nxt = unit_q.get_nowait()
                            except queue.Empty:
                                break
                            if nxt is None:
                                self.vocoder.abort_session(voc_sess)
                                return
                            batch_u.extend(nxt[0])
                            if nxt[1]:
                                saw_end = True

                        if abort_flag is not None and abort_flag.is_set():
                            self.vocoder.abort_session(voc_sess)
                            break

                        pcm_f32 = self.vocoder.push_units(batch_u, voc_sess, finalize=saw_end, min_chunk_units=17)
                        if self.voc_cuda_stream is not None:
                            self.voc_cuda_stream.synchronize()
                        if len(pcm_f32) > 0:
                            if first_audio_box[0] is None:
                                first_audio_box[0] = (time.perf_counter() - t_start) * 1000.0
                            pcm_i16 = np.clip(pcm_f32 * 32767.0, -32768, 32767).astype(np.int16).tobytes()
                            event_q.put(("audio_pcm", pcm_i16, first_audio_box[0]))
                        if saw_end:
                            break
            finally:
                voc_done_evt.set()

        def _thinker_talker_worker() -> None:
            raw_full = ""
            clean_spoken = ""
            aborted_or_short = False
            try:
                with self.lock, torch.inference_mode():
                    if self.seq_len > 2800:
                        self.reset_context()

                    if trigger_mode == "tool_obs" and tool_obs is not None:
                        chunk_ids = self.enc(f"<|turn>tool\n{tool_obs}<turn|>\n<|turn>model\n")
                    elif trigger_mode == "text" and text_query is not None:
                        chunk_ids = self.enc(f"<|turn>user\n{text_query}<turn|>\n<|turn>model\n")
                    else:
                        chunk_ids = self.enc("<|turn>user\n<unused0><turn|>\n<|turn>model\n")

                    inp = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
                    m_out = self.thinker.model(input_ids=inp, past_key_values=self.cache, use_cache=True, return_dict=True)
                    self.cache = m_out.past_key_values
                    self.seq_len += len(chunk_ids)

                    last_h = m_out.last_hidden_state[:, -1:, :]
                    logits_1 = self.thinker.lm_head(last_h)
                    pred_h = last_h[0, 0, :]
                    next_id = int(logits_1[0, 0].argmax().item())

                    # Short-circuit if Thinker decides not to speak (<unused1>, <unused4>, <unused5>)
                    if next_id in (7, 10, 11):
                        tail = [next_id, self.turn_end_id, self.newline_id]
                        t_inp = torch.tensor([tail], dtype=torch.long, device=self.device)
                        t_out = self.thinker.model(input_ids=t_inp, past_key_values=self.cache, use_cache=True, return_dict=True)
                        self.cache = t_out.past_key_values
                        self.seq_len += len(tail)
                        unit_q.put(None)
                        aborted_or_short = True
                        event_q.put(("state_only", self.tokenizer.decode([next_id], skip_special_tokens=False)))
                        return

                    talker_sess = TalkerStreamSession()
                    gen_ids: List[int] = []
                    emitted_tool_count = 0

                    for _ in range(96):
                        if abort_flag is not None and abort_flag.is_set():
                            talker_sess.reset()
                            unit_q.put(None)
                            close_ids = [self.turn_end_id, self.newline_id]
                            c_inp = torch.tensor([close_ids], dtype=torch.long, device=self.device)
                            c_out = self.thinker.model(input_ids=c_inp, past_key_values=self.cache, use_cache=True, return_dict=True)
                            self.cache = c_out.past_key_values
                            self.seq_len += 2
                            aborted_or_short = True
                            return

                        is_end = (next_id == self.turn_end_id or next_id == self.tokenizer.eos_token_id)
                        u_chunk = self.talker.step_gemma_token(
                            self.tokenizer,
                            next_id,
                            pred_h,
                            talker_sess,
                            is_turn_end=is_end,
                            temperature=0.55,
                        )
                        if u_chunk or is_end:
                            unit_q.put((u_chunk, is_end))

                        if is_end:
                            break

                        gen_ids.append(next_id)
                        if next_id == self.tc_end_id:
                            raw_so_far = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                            _, calls = parse_gemma4_tool_calls(raw_so_far)
                            while emitted_tool_count < len(calls):
                                event_q.put(("tool_call", calls[emitted_tool_count]))
                                emitted_tool_count += 1

                        s_inp = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                        s_out = self.thinker.model(input_ids=s_inp, past_key_values=self.cache, use_cache=True, return_dict=True)
                        self.cache = s_out.past_key_values
                        self.seq_len += 1
                        last_h = s_out.last_hidden_state[:, -1:, :]
                        logits_1 = self.thinker.lm_head(last_h)
                        pred_h = last_h[0, 0, :]
                        next_id = int(logits_1[0, 0].argmax().item())
                    else:
                        # Reached 96-token cap without <turn|>: ensure vocoder receives finalize!
                        u_tail = self.talker.step_gemma_token(
                            self.tokenizer, self.turn_end_id, pred_h, talker_sess, is_turn_end=True, temperature=0.55
                        )
                        unit_q.put((u_tail, True))

                    end_tail = [self.turn_end_id, self.newline_id]
                    e_inp = torch.tensor([end_tail], dtype=torch.long, device=self.device)
                    e_out = self.thinker.model(input_ids=e_inp, past_key_values=self.cache, use_cache=True, return_dict=True)
                    self.cache = e_out.past_key_values
                    self.seq_len += 2

                    raw_full = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                    clean_spoken, _ = parse_gemma4_tool_calls(raw_full)

                # Wait outside self.lock for final Vocoder chunk so all audio_pcm arrive strictly before 'done'
                voc_done_evt.wait(timeout=15.0)
                if not aborted_or_short and not (abort_flag is not None and abort_flag.is_set()):
                    event_q.put(("done", raw_full, clean_spoken, first_audio_box[0] or 0.0))
            finally:
                unit_q.put(None)
                voc_done_evt.wait(timeout=5.0)
                event_q.put(("__eof__",))

        threading.Thread(target=_vocoder_worker, daemon=True).start()
        threading.Thread(target=_thinker_talker_worker, daemon=True).start()

        while True:
            ev = event_q.get()
            if ev[0] == "__eof__":
                break
            yield ev


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gemma-4-E2B Route 2 (II-A) 三段式原生流式全双工语音对话</title>
<style>
  body { background:#0b1120; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; margin:0; padding:24px; }
  .wrap { max-width:900px; margin:0 auto; }
  .card { background:#111827; border:1px solid #1f2937; border-radius:16px; padding:20px; margin-bottom:16px; box-shadow:0 8px 24px rgba(0,0,0,0.35); }
  h1 { margin:0 0 6px; font-size:21px; color:#f8fafc; }
  .sub { color:#94a3b8; font-size:13px; margin-bottom:14px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  button { background:#2563eb; color:#fff; border:none; border-radius:10px; padding:10px 16px; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#1d4ed8; }
  button.stop { background:#dc2626; }
  button.sec { background:#334155; }
  .badge { padding:5px 12px; border-radius:999px; font-size:12px; font-weight:700; background:#1e293b; color:#38bdf8; }
  .ttfa { color:#4ade80; font-weight:700; font-size:14px; }
  .meter { height:8px; background:#1e293b; border-radius:4px; overflow:hidden; margin-top:12px; }
  .meter-bar { height:100%; width:0%; background:linear-gradient(90deg,#22c55e,#38bdf8); transition:width 80ms linear; }
  #log { height:380px; overflow-y:auto; display:flex; flex-direction:column; gap:10px; padding-right:6px; }
  .msg { padding:10px 14px; border-radius:12px; font-size:14px; line-height:1.5; max-width:85%; }
  .msg.ai { background:#1e293b; align-self:flex-start; border-left:3px solid #22c55e; }
  .msg.sys { background:#0f172a; color:#94a3b8; font-size:12px; align-self:center; border:1px solid #1e293b; }
  .msg.tool { background:#1c1917; color:#fbbf24; font-family:monospace; font-size:12px; align-self:flex-start; border-left:3px solid #f59e0b; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>⚡ Gemma-4-E2B Route 2 (II-A) 谷歌三段式原生流式全双工语音引擎</h1>
    <div class="sub">Gemma-4-E2B Thinker (全双工槽位 + 零ASR工具) ──► Hidden States Gate-Fusion Talker (25Hz) ──► CosyVoice2 GPU Vocoder (24kHz 直出)</div>
    <div class="row">
      <button id="micBtn" onclick="toggleMic()">🎙️ 开启麦克风对话</button>
      <button id="modeBtn" class="sec" onclick="toggleDuplexMode()">🔊 外放防回声模式 (默认)</button>
      <button class="sec" onclick="resetSession()">🔄 重置上下文</button>
      <span id="stateBadge" class="badge">状态: 空闲待命</span>
      <span id="ttfaLabel" class="ttfa">首包音频延迟 (TTFA): -- ms</span>
    </div>
    <div class="meter"><div id="meterBar" class="meter-bar"></div></div>
    <div class="row" style="margin-top:12px;">
      <span style="font-size:12px;color:#94a3b8;">一键免麦测试:</span>
      <button class="sec" onclick="sendTextSim('你好呀！你叫什么名字？')">👋 测试问候 (挂9工具不误触发)</button>
      <button class="sec" onclick="sendTextSim('帮我查一下明天北京的天气怎么样')">🌤️ 测试异步查天气</button>
      <button class="sec" onclick="bargeInSim()">🛑 立即打断清播报</button>
    </div>
  </div>
  <div class="card">
    <div id="log"></div>
  </div>
</div>
<script>
let ws = null;
let audioCtx = null;
let micStream = null;
let scriptNode = null;
let micActive = false;
let duplexMode = 'speaker';
let nextPlayTime = 0;
let activeSources = [];
let isAiPlaying = false;
let lastPlayEndMs = 0;
let isFirstChunkOfTurn = true;

function addLog(cls, text) {
  const el = document.getElementById('log');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function stopAllPlayback() {
  for (const s of activeSources) {
    try { s.stop(0); s.disconnect(); } catch (e) {}
  }
  activeSources = [];
  nextPlayTime = 0;
  isFirstChunkOfTurn = true;
  if (isAiPlaying) lastPlayEndMs = performance.now();
  isAiPlaying = false;
}

function playPcm24kBase64(b64) {
  ensureAudioCtx();
  const bin = atob(b64);
  const len = bin.length / 2;
  const f32 = new Float32Array(len);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < len; i++) {
    f32[i] = view.getInt16(i * 2, true) / 32768.0;
  }
  const buf = audioCtx.createBuffer(1, len, 24000);
  buf.getChannelData(0).set(f32);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (isFirstChunkOfTurn) {
    // 320ms jitter cushion + 520ms Chunk-0 audio = 840ms continuous runway so subsequent 800ms chunks never underrun
    nextPlayTime = Math.max(nextPlayTime, now + 0.32);
    isFirstChunkOfTurn = false;
  } else if (nextPlayTime < now + 0.01) {
    nextPlayTime = now + 0.05;
  }
  src.start(nextPlayTime);
  nextPlayTime += buf.duration;
  isAiPlaying = true;
  activeSources.push(src);
  src.onended = () => {
    activeSources = activeSources.filter(x => x !== src);
    if (activeSources.length === 0) {
      isAiPlaying = false;
      lastPlayEndMs = performance.now();
    }
  };
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onopen = () => {
    addLog('sys', '✅ 已连接 GPU 三段式流式语音服务器');
    ws.send(JSON.stringify({ type: 'set_mode', mode: duplexMode }));
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'state') {
      document.getElementById('stateBadge').textContent = '状态: ' + msg.state + (msg.ms ? ` (${msg.ms.toFixed(0)}ms)` : '');
    } else if (msg.type === 'stop_audio') {
      stopAllPlayback();
      addLog('sys', '🛑 检测到真人打断 (<unused6>)，0ms 清空声码器与播放器队列！');
    } else if (msg.type === 'audio_chunk') {
      if (msg.ttfa_ms) {
        document.getElementById('ttfaLabel').textContent = `首包音频延迟 (TTFA): ${msg.ttfa_ms.toFixed(0)} ms`;
      }
      playPcm24kBase64(msg.pcm_b64);
    } else if (msg.type === 'tool_call') {
      addLog('tool', `🔧 异步调用工具: ${msg.call.name}(${JSON.stringify(msg.call.args)})`);
    } else if (msg.type === 'assistant_done') {
      isFirstChunkOfTurn = true;
      if (msg.text) addLog('ai', `🤖 ${msg.text}  [TTFA: ${msg.ttfa_ms.toFixed(0)}ms]`);
    } else if (msg.type === 'info') {
      addLog('sys', msg.text);
    }
  };
  ws.onclose = () => setTimeout(connectWs, 1500);
}
connectWs();

function toggleDuplexMode() {
  duplexMode = (duplexMode === 'speaker') ? 'headset' : 'speaker';
  const btn = document.getElementById('modeBtn');
  if (duplexMode === 'speaker') {
    btn.textContent = '🔊 外放防回声模式 (默认)';
    btn.style.background = '#334155';
  } else {
    btn.textContent = '🎧 耳机全双工自由打断';
    btn.style.background = '#0d9488';
  }
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ type: 'set_mode', mode: duplexMode }));
  }
}

async function toggleMic() {
  ensureAudioCtx();
  const btn = document.getElementById('micBtn');
  if (micActive) {
    micActive = false;
    if (scriptNode) scriptNode.disconnect();
    if (micStream) micStream.getTracks().forEach(t => t.stop());
    btn.textContent = '🎙️ 开启麦克风对话';
    btn.className = '';
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 }
    });
  } catch (err) {
    addLog('sys', '❌ 麦克风权限获取失败: ' + err.message + ' (请使用 HTTPS cloudflared 链接在 Chrome 打开)');
    return;
  }
  const src = audioCtx.createMediaStreamSource(micStream);
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  const muteGain = audioCtx.createGain();
  muteGain.gain.value = 0;
  let ring = new Float32Array(0);

  scriptNode.onaudioprocess = (e) => {
    if (!micActive || !ws || ws.readyState !== 1) return;
    const rawInp = e.inputBuffer.getChannelData(0);
    let inp = rawInp;
    if (audioCtx.sampleRate !== 16000) {
      const ratio = audioCtx.sampleRate / 16000;
      const newLen = Math.round(rawInp.length / ratio);
      inp = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        const idx = i * ratio;
        const i0 = Math.floor(idx);
        const i1 = Math.min(i0 + 1, rawInp.length - 1);
        const frac = idx - i0;
        inp[i] = rawInp[i0] * (1 - frac) + rawInp[i1] * frac;
      }
    }
    const merged = new Float32Array(ring.length + inp.length);
    merged.set(ring, 0);
    merged.set(inp, ring.length);
    ring = merged;

    while (ring.length >= 7680) {
      const frame = ring.slice(0, 7680);
      ring = ring.slice(7680);
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      const rms = Math.sqrt(sum / frame.length);
      document.getElementById('meterBar').style.width = Math.min(100, rms * 450) + '%';

      const inTailCooldown = (performance.now() - lastPlayEndMs) < 700;
      const i16 = new Int16Array(7680);
      for (let i = 0; i < 7680; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      const u8 = new Uint8Array(i16.buffer);
      let bin = '';
      for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
      ws.send(JSON.stringify({
        type: 'mic_frame',
        pcm16_b64: btoa(bin),
        rms: rms,
        ai_playing: (isAiPlaying || inTailCooldown)
      }));
    }
  };
  src.connect(scriptNode);
  scriptNode.connect(muteGain);
  muteGain.connect(audioCtx.destination);
  micActive = true;
  btn.textContent = '⏹️ 关闭麦克风';
  btn.className = 'stop';
  addLog('sys', '🎙️ 麦克风已启动 (默认「🔊外放防回声」不会自激；若戴耳机可点上方按钮切换「🎧耳机全双工自由打断」)');
}

function sendTextSim(txt) {
  ensureAudioCtx();
  if (ws && ws.readyState === 1) {
    addLog('sys', '🗣️ 模拟输入: ' + txt);
    ws.send(JSON.stringify({ type: 'text_sim', text: txt }));
  }
}
function bargeInSim() {
  stopAllPlayback();
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'barge_sim' }));
}
function resetSession() {
  stopAllPlayback();
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'reset' }));
}
</script>
</body>
</html>
"""


def is_human_speech_frame(wav_7680: np.ndarray, rms: float, noise_floor: float, already_speaking: bool) -> bool:
    """0.04ms spectral + adaptive SNR check blocking stationary fan noise (<0.48 vband) and desk thumps (<0.05 vband)."""
    min_rms = max(0.015, noise_floor * 1.7) if already_speaking else max(0.022, noise_floor * 2.3)
    if rms < min_rms:
        return False
    spec = np.abs(np.fft.rfft(wav_7680)) ** 2
    tot = float(np.sum(spec) + 1e-12)
    # 7680 samples @ 16kHz -> bin spacing = 2.0833 Hz; 150..3800 Hz = bins 72..1824
    vband_ratio = float(np.sum(spec[72:1825])) / tot
    return vband_ratio >= (0.50 if already_speaking else 0.56)


def create_app(engine: Route2OmniEngine) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()

        duplex_mode = "speaker"  # "speaker" (echo-safe default) or "headset" (free full-duplex barge-in)
        valid_speech_frames = 0
        silence_frames = 0
        barge_frames = 0
        noise_floor = 0.008
        is_generating = False
        server_play_until = 0.0
        cooldown_until = 0.0
        abort_event = threading.Event()
        active_tool_tasks: Dict[str, asyncio.Task] = {}

        async def send_json(payload: Dict[str, Any]) -> None:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))

        async def run_streaming_turn(trigger_mode: str, tool_obs: Optional[str] = None, text_query: Optional[str] = None) -> None:
            nonlocal is_generating, server_play_until, cooldown_until
            is_generating = True
            abort_event.clear()
            q: asyncio.Queue = asyncio.Queue()

            def _producer() -> None:
                try:
                    for item in engine.stream_response_generator(
                        trigger_mode=trigger_mode,
                        tool_obs=tool_obs,
                        text_query=text_query,
                        abort_flag=abort_event,
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, item)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, ("__eof__",))

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                ev = await q.get()
                kind = ev[0]
                if kind == "__eof__":
                    break
                if kind == "state_only":
                    await send_json({"type": "state", "state": ev[1], "ms": 0.0})
                elif kind == "audio_pcm":
                    pcm_bytes, ttfa_ms = ev[1], ev[2]
                    chunk_dur = (len(pcm_bytes) / 2.0) / float(SR_OUT)
                    now_m = time.monotonic()
                    server_play_until = max(now_m, server_play_until) + chunk_dur
                    cooldown_until = server_play_until + 0.75
                    await send_json({
                        "type": "audio_chunk",
                        "pcm_b64": base64.b64encode(pcm_bytes).decode("ascii"),
                        "ttfa_ms": ttfa_ms,
                    })
                elif kind == "tool_call":
                    call_spec = ev[1]
                    await send_json({"type": "tool_call", "call": call_spec})

                    async def _bg_tool(cs: Dict[str, Any]) -> None:
                        res = await execute_async_tool(cs["name"], cs["args"])
                        if abort_event.is_set():
                            return
                        obs_json = json.dumps(res, ensure_ascii=False)
                        while is_generating and not abort_event.is_set():
                            await asyncio.sleep(0.05)
                        if not abort_event.is_set():
                            await run_streaming_turn("tool_obs", tool_obs=obs_json)

                    t = asyncio.create_task(_bg_tool(call_spec))
                    active_tool_tasks[call_spec["id"]] = t
                elif kind == "done":
                    raw_full, clean_spoken, ttfa_ms = ev[1], ev[2], ev[3]
                    state_m = re.match(r"(<unused\d+>)", raw_full)
                    st = state_m.group(1) if state_m else "<unused2>"
                    await send_json({"type": "state", "state": st, "ms": ttfa_ms})
                    await send_json({"type": "assistant_done", "text": clean_spoken, "ttfa_ms": ttfa_ms})

            is_generating = False

        try:
            while True:
                raw_msg = await ws.receive_text()
                data = json.loads(raw_msg)
                mtype = data.get("type")

                if mtype == "set_mode":
                    duplex_mode = "headset" if data.get("mode") == "headset" else "speaker"
                    label = "🎧 耳机全双工自由打断模式" if duplex_mode == "headset" else "🔊 外放防回声安全模式"
                    await send_json({"type": "info", "text": f"已切换至: {label}"})

                elif mtype == "reset":
                    abort_event.set()
                    for t in active_tool_tasks.values():
                        t.cancel()
                    active_tool_tasks.clear()
                    await loop.run_in_executor(None, engine.reset_context)
                    valid_speech_frames = 0
                    silence_frames = 0
                    barge_frames = 0
                    server_play_until = 0.0
                    cooldown_until = 0.0
                    await send_json({"type": "info", "text": "🔄 会话 KV Cache 与声码器状态已重置"})

                elif mtype == "barge_sim":
                    abort_event.set()
                    for t in active_tool_tasks.values():
                        t.cancel()
                    active_tool_tasks.clear()
                    valid_speech_frames = 0
                    silence_frames = 0
                    server_play_until = 0.0
                    cooldown_until = time.monotonic() + 0.35
                    await send_json({"type": "stop_audio"})
                    await send_json({"type": "state", "state": "<unused6>", "ms": 0.5})

                elif mtype == "text_sim":
                    abort_event.set()
                    valid_speech_frames = 0
                    silence_frames = 0
                    await asyncio.sleep(0.02)
                    sim_text = str(data["text"])

                    async def _run_audio_sim(q_txt: str) -> None:
                        import io
                        import av
                        import edge_tts

                        comm = edge_tts.Communicate(q_txt, "zh-CN-YunxiNeural")
                        mp3_buf = bytearray()
                        async for chunk in comm.stream():
                            if chunk["type"] == "audio":
                                mp3_buf.extend(chunk["data"])
                        container = av.open(io.BytesIO(bytes(mp3_buf)))
                        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR_IN)
                        arrs = []
                        for fr in container.decode(audio=0):
                            for ofr in resampler.resample(fr):
                                arrs.append(ofr.to_ndarray()[0])
                        for ofr in resampler.resample(None):
                            arrs.append(ofr.to_ndarray()[0])
                        wav_sim = np.concatenate(arrs).astype(np.float32) if arrs else np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                        win = 400
                        nf = len(wav_sim) // win
                        if nf > 2:
                            f_rms = np.sqrt(np.mean(wav_sim[: nf * win].reshape(nf, win) ** 2, axis=1))
                            nz = np.where(f_rms > 0.010)[0]
                            if len(nz) > 0:
                                wav_sim = wav_sim[max(0, (nz[0] - 1) * win) : min(len(wav_sim), (nz[-1] + 2) * win)]
                        for start in range(0, len(wav_sim), CHUNK_SAMPLES):
                            seg = wav_sim[start : start + CHUNK_SAMPLES]
                            if start > 0 and len(seg) < 1600:
                                break
                            if len(seg) < CHUNK_SAMPLES:
                                seg = np.pad(seg, (0, CHUNK_SAMPLES - len(seg)))
                            st, ms = await loop.run_in_executor(None, engine.prefill_audio_tick, seg, False)
                            await send_json({"type": "state", "state": st, "ms": ms})
                        await run_streaming_turn("silence")

                    asyncio.create_task(_run_audio_sim(sim_text))

                elif mtype == "mic_frame":
                    pcm_bytes = base64.b64decode(data["pcm16_b64"])
                    wav = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(data.get("rms", 0.0))
                    ai_playing = bool(data.get("ai_playing", False))
                    now_m = time.monotonic()

                    if ai_playing:
                        cooldown_until = max(cooldown_until, now_m + 0.70)

                    in_playback_window = is_generating or ai_playing or (now_m < server_play_until) or (now_m < cooldown_until)

                    # Case 1: AI is speaking or within post-playback room-reverb cooldown
                    if in_playback_window:
                        valid_speech_frames = 0
                        silence_frames = 0
                        if duplex_mode == "speaker":
                            # Echo-safe speakerphone mode: prevent speaker bleed from self-looping
                            continue

                        # Headset full-duplex mode: verify real human barge-in on Thinker without speaker bleed
                        if rms >= 0.028:
                            barge_frames += 1
                            if not is_generating:
                                st, ms = await loop.run_in_executor(None, engine.prefill_audio_tick, wav, True)
                                if "<unused6>" in st or "<unused1>" in st:
                                    for t in active_tool_tasks.values():
                                        t.cancel()
                                    active_tool_tasks.clear()
                                    server_play_until = 0.0
                                    cooldown_until = 0.0
                                    await send_json({"type": "stop_audio"})
                                    await send_json({"type": "state", "state": st, "ms": ms})
                                    valid_speech_frames = 1
                                    silence_frames = 0
                                    barge_frames = 0
                            elif barge_frames >= 2:
                                abort_event.set()
                                while is_generating:
                                    await asyncio.sleep(0.005)
                                st, ms = await loop.run_in_executor(None, engine.prefill_audio_tick, wav, True)
                                if "<unused6>" in st or "<unused1>" in st:
                                    for t in active_tool_tasks.values():
                                        t.cancel()
                                    active_tool_tasks.clear()
                                    server_play_until = 0.0
                                    cooldown_until = 0.0
                                    await send_json({"type": "stop_audio"})
                                    await send_json({"type": "state", "state": "<unused6><unused1>", "ms": ms})
                                    valid_speech_frames = 1
                                    silence_frames = 0
                                barge_frames = 0
                        else:
                            barge_frames = 0
                        continue

                    # Case 2: Normal listening & state-verified endpointing
                    barge_frames = 0
                    if is_human_speech_frame(wav, rms, noise_floor, already_speaking=(valid_speech_frames > 0)):
                        st, ms = await loop.run_in_executor(
                            None, engine.prefill_audio_tick, wav, (valid_speech_frames == 0)
                        )
                        await send_json({"type": "state", "state": st, "ms": ms})
                        if "<unused1>" in st or "<unused6>" in st:
                            valid_speech_frames += 1
                            silence_frames = 0
                        elif valid_speech_frames > 0:
                            silence_frames += 1
                            if silence_frames >= 1:
                                valid_speech_frames = 0
                                silence_frames = 0
                                cooldown_until = now_m + 0.50
                                asyncio.create_task(run_streaming_turn("silence"))
                    else:
                        if valid_speech_frames == 0 and rms < 0.035:
                            noise_floor = float(min(0.025, 0.90 * noise_floor + 0.10 * rms))
                        if valid_speech_frames >= 1:
                            silence_frames += 1
                            if silence_frames >= 1:
                                valid_speech_frames = 0
                                silence_frames = 0
                                cooldown_until = now_m + 0.50
                                asyncio.create_task(run_streaming_turn("silence"))
        except WebSocketDisconnect:
            abort_event.set()

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--stage1_adapter", type=str, default="")
    parser.add_argument("--stage2_talker", type=str, default="")
    parser.add_argument("--prompt_wav", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    engine = Route2OmniEngine(
        base_model=args.base_model,
        stage1_adapter=args.stage1_adapter,
        stage2_talker=args.stage2_talker,
        prompt_wav=args.prompt_wav,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
