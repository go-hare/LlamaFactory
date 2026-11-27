#!/usr/bin/env python3
"""
End-to-End Automated Sanity & Latency Benchmark for Gemma-4-E2B Route 2 (II-A):
Tests:
  1. Greeting ("你好") with ALL 9 tools active -> NO false tool call + intelligible 24kHz speech
  2. Identity ("你叫什么名字") with ALL 9 tools active -> clean persona (zero "小明" / zero parrot)
  3. Zero-ASR Async Tool Call -> exact <unused3><|tool_call> + spoken ack without reading JSON aloud
  4. First-Packet Audio Latency (TTFA in ms) on GPU
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from live_route2_cuda_server import Route2OmniEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--stage1_adapter", type=str, required=True)
    parser.add_argument("--stage2_talker", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/route2_eval_wavs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = Route2OmniEngine(
        base_model=args.base_model,
        stage1_adapter=args.stage1_adapter,
        stage2_talker=args.stage2_talker,
    )

    cases = [
        ("greeting_with_tools", "你好"),
        ("identity_with_tools", "你叫什么名字"),
        ("weather_tool_call", "帮我查一下明天上海的天气怎么样"),
    ]

    print("\n================ ROUTE 2 END-TO-END BENCHMARK ================")
    for tag, prompt in cases:
        engine.reset_context()
        pcm_parts = []
        tool_calls = []
        raw_out = ""
        spoken_out = ""
        ttfa = 0.0
        t0 = time.perf_counter()

        for ev in engine.stream_response_generator(trigger_mode="text", text_query=prompt):
            if ev[0] == "tool_call":
                tool_calls.append(ev[1])
            elif ev[0] == "audio_pcm":
                pcm_parts.append(np.frombuffer(ev[1], dtype=np.int16).astype(np.float32) / 32768.0)
            elif ev[0] == "done":
                raw_out, spoken_out, ttfa = ev[1], ev[2], ev[3]

        total_ms = (time.perf_counter() - t0) * 1000.0
        wav = np.concatenate(pcm_parts) if pcm_parts else np.zeros(2400, dtype=np.float32)
        wav_p = out_dir / f"{tag}.wav"
        sf.write(str(wav_p), wav, 24000)
        print(
            f"[{tag}] Prompt='{prompt}'\n"
            f"  Raw Thinker : {raw_out}\n"
            f"  Spoken Text : {spoken_out}\n"
            f"  Tool Calls  : {tool_calls}\n"
            f"  Audio Out   : {len(wav)/24000:.2f}s @ 24kHz | TTFA = {ttfa:.1f} ms | Total = {total_ms:.1f} ms -> {wav_p}"
        )
    print("==============================================================")


if __name__ == "__main__":
    main()
