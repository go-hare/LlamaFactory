#!/usr/bin/env python3
"""
LlamaFactory 路线一 (Gemma-4-E2B/E4B 全双工原生音频 + 异步工具调用) 数据集构造脚本

直接兼容本仓库两种原始数据格式：
1. 本仓库 Gemma-4 预格式化 JSONL (如 data/real_conversion_pilot_v1/train.jsonl, data/diffusion_600m/train.jsonl)
   - 自动解析 `<|turn>system` / `<|turn>user` / `<|turn>model`
   - 自动剥离耗时的 `<|channel>thought...<channel|>` 思维链，确保语音首包零延迟
2. 标准 ShareGPT / Glaive 工具调用 JSON (如 data/glaive_toolcall_zh_demo.json)
   - 自动转换 `function_call` 为 Gemma-4 原生 `<|tool_call>call:name{...}<tool_call|>`
   - 自动注入 4 大全双工异步工具状态：
     * 非阻塞垫话 + `<|tool_call>`
     * `<unused3>` (`waiting_tool` 后台异步等待态)
     * `<unused6><unused7>c1` (工具未返回时用户改口作废旧任务)
     * `WHEN_IDLE` (工具结果返回时若用户正在说话，先输出 `<unused1>` 压住不抢话)
3. 混入真实 16kHz 音频切片 (`<audio>` 占位符 + `audios` 列表，配合 `template: gemma4n`)：
   - 混入咳嗽/低能量背景杂音帧 -> 监督 `<unused5>` (`ignore_noise` 主动抗噪不断流)
   - 混入真实语音分片与停顿   -> 监督 `<unused1>` (`user_talking` 韵律防抢话)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

SLOTS = {
    "no_voice": "<unused0>",       # ID 6
    "user_talking": "<unused1>",   # ID 7
    "start_speak": "<unused2>",    # ID 8
    "waiting_tool": "<unused3>",   # ID 9
    "idle_think": "<unused4>",     # ID 10
    "ignore_noise": "<unused5>",   # ID 11
    "interrupted": "<unused6>",    # ID 12
    "cancel_tool": "<unused7>",    # ID 13
}

FILLERS_ZH = ["好的，马上帮您查。", "稍等，我看一下。", "明白，正在查询。"]
BACKCHANNELS_ZH = ["嗯嗯", "好的", "对", "嗯"]
THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
TURN_RE = re.compile(r"<\|turn>(system|user|model|tool)\n(.*?)<turn\|>", re.DOTALL)


def split_speech_chunks(text: str, min_chars: int = 3, max_chars: int = 6) -> List[str]:
    """按 0.48s 真人正常语速 (~4字/0.48s) 切分文本，避免未播先知假历史污染"""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        step = random.randint(min_chars, max_chars)
        chunk = text[i : i + step]
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def format_gemma4_tool_call(call_id: str, func_name: str, args_obj: Dict[str, Any]) -> str:
    """生成 Gemma-4 原生 `<|tool_call>call:name{id:<|"|>c1<|"|>,...}<tool_call|>` 字符串"""
    def _fmt_val(v: Any) -> str:
        if isinstance(v, str):
            return f'<|"|>{v}<|"|>'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ",".join(_fmt_val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{_fmt_val(val)}" for k, val in sorted(v.items())) + "}"
        return f'<|"|>{str(v)}<|"|>'

    merged = {"id": call_id, **args_obj}
    body = ",".join(f"{k}:{_fmt_val(v)}" for k, v in merged.items())
    return f"<|tool_call>call:{func_name}{{{body}}}<tool_call|>"


def parse_gemma4_raw_row(row: Dict[str, Any]) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """解析本仓库 real_conversion_pilot_v1 / diffusion_600m 的 prompt + response 格式"""
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    if not prompt or not response:
        return None

    clean_resp = THOUGHT_RE.sub("", response).replace("<turn|>", "").strip()
    # 去掉过长的角色动作括号或代码块（语音对话不宜念大段代码）
    if "```" in clean_resp or len(clean_resp) > 220 or not clean_resp:
        return None

    turns = TURN_RE.findall(prompt)
    sys_text = ""
    qa_pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None

    for role, content in turns:
        c = content.replace("<|think|>", "").strip()
        if role == "system":
            sys_text = c
        elif role == "user":
            pending_user = c
        elif role == "model" and pending_user is not None:
            qa_pairs.append((pending_user, THOUGHT_RE.sub("", c).strip()))
            pending_user = None

    # 最后一个 user turn 对应 row["response"]
    last_user_match = re.findall(r"<\|turn>user\n(.*?)<turn\|>\s*<\|turn>model\n\s*$", prompt, re.DOTALL)
    if last_user_match:
        qa_pairs.append((last_user_match[-1].strip(), clean_resp))

    if not qa_pairs:
        return None
    return sys_text, qa_pairs


def build_chat_duplex_entry(
    sys_text: str,
    qa_pairs: List[Tuple[str, str]],
    noise_wav_rel: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """将多轮纯文本问答切分为 ShareGPT 全双工微回合（可选混入真实咳嗽/背景杂音音频块）"""
    convs: List[Dict[str, str]] = []
    audios: List[str] = []

    for idx, (u_text, a_text) in enumerate(qa_pairs):
        u_chunks = split_speech_chunks(u_text, 3, 7)
        if not u_chunks:
            continue

        # 1. 用户分片输入 + 15% 概率模拟句中喘气停顿
        for c_i, uc in enumerate(u_chunks):
            convs.append({"from": "human", "value": uc})
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            if c_i < len(u_chunks) - 1 and random.random() < 0.15:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        # 2. AI 按 0.48s 真实语速 (3~6字) 流式吐出回答
        a_chunks = split_speech_chunks(a_text, 4, 7)[:12]
        if not a_chunks:
            continue

        will_interrupt = idx + 1 < len(qa_pairs) and len(a_chunks) >= 3 and random.random() < 0.35
        stop_at = random.randint(1, len(a_chunks) - 1) if will_interrupt else len(a_chunks)

        for a_i in range(stop_at):
            ac = a_chunks[a_i]
            if a_i == 0:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ac}"})
            else:
                r = random.random()
                if noise_wav_rel and r < 0.12 and len(audios) < 2:
                    # 混入真实咳嗽/环境杂音音频块 (<audio>) -> 监督输出 <unused5> 抗噪不断流！
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(noise_wav_rel)
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                elif r < 0.20:
                    convs.append({"from": "human", "value": random.choice(BACKCHANNELS_ZH)})
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": ac})

        if will_interrupt:
            # 下一轮用户的首片直接插进来触发真打断 <unused6>
            next_u_first = split_speech_chunks(qa_pairs[idx + 1][0], 3, 7)[0]
            convs.append({"from": "human", "value": next_u_first})
            convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
        else:
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

    if len(convs) < 4:
        return None

    entry: Dict[str, Any] = {"conversations": convs, "tools": "[]"}
    if sys_text:
        entry["system"] = sys_text
    if audios:
        entry["audios"] = audios
    return entry


def build_tool_duplex_entries(raw_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将 ShareGPT 工具调用样本转换为全双工异步工具状态机样本（含正常异步、中途改口取消、WHEN_IDLE 延迟播报）"""
    src_convs = raw_item.get("conversations", [])
    tools_str = raw_item.get("tools", "[]")
    if len(src_convs) < 4 or src_convs[1].get("from") != "function_call":
        return []

    u_query = src_convs[0]["value"].strip()
    fc_raw = json.loads(src_convs[1]["value"])
    obs_raw = src_convs[2]["value"].strip()
    final_ans = src_convs[3]["value"].strip()

    func_name = fc_raw["name"]
    func_args = fc_raw.get("arguments", {})
    if isinstance(func_args, str):
        func_args = json.loads(func_args)

    results: List[Dict[str, Any]] = []
    u_chunks = split_speech_chunks(u_query, 4, 8)
    ans_chunks = split_speech_chunks(final_ans, 4, 8)[:10]
    if not u_chunks or not ans_chunks:
        return []

    # 形态 1：NON_BLOCKING 垫话发起工具 + <unused3> 后台等待 + 分片播报结果
    c1_convs: List[Dict[str, str]] = []
    for uc in u_chunks:
        c1_convs.append({"from": "human", "value": uc})
        c1_convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

    call_str = format_gemma4_tool_call("c1", func_name, func_args)
    filler = random.choice(FILLERS_ZH)
    c1_convs.append({"from": "human", "value": SLOTS["no_voice"]})
    c1_convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{filler}{call_str}"})

    # 模拟后台工具耗时 1~2 个 0.48s Tick -> 输出 <unused3> (waiting_tool)
    for _ in range(random.randint(1, 2)):
        c1_convs.append({"from": "human", "value": SLOTS["no_voice"]})
        c1_convs.append({"from": "gpt", "value": SLOTS["waiting_tool"]})

    # 30% 概率模拟 WHEN_IDLE (工具返回时用户恰好开口补充半句话 -> 先输出 <unused1> 压住不抢话！)
    if random.random() < 0.30:
        c1_convs.append({"from": "observation", "value": f"{obs_raw}\n[用户同时正在补充说话: 顺便简单说重点就行]"})
        c1_convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
        c1_convs.append({"from": "human", "value": SLOTS["no_voice"]})
        c1_convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ans_chunks[0]}"})
    else:
        c1_convs.append({"from": "observation", "value": obs_raw})
        c1_convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ans_chunks[0]}"})

    for ac in ans_chunks[1:]:
        c1_convs.append({"from": "human", "value": SLOTS["no_voice"]})
        c1_convs.append({"from": "gpt", "value": ac})
    c1_convs.append({"from": "human", "value": SLOTS["no_voice"]})
    c1_convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

    results.append({"conversations": c1_convs, "tools": tools_str, "audios": []})

    # 形态 2 (以 35% 概率额外构造一条)：工具尚未返回时用户突然改口 -> <unused6><unused7>c1 取消任务！
    if random.random() < 0.35:
        cancel_convs = list(c1_convs[: len(u_chunks) * 2 + 2])  # 截取到刚发出 c1
        cancel_convs.append({"from": "human", "value": "算了先不用查这个了"})
        cancel_convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
        cancel_convs.append({"from": "human", "value": SLOTS["no_voice"]})
        cancel_convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已为您取消查询。"})
        results.append({"conversations": cancel_convs, "tools": tools_str, "audios": []})

    return results


def generate_real_audio_clips(audio_out_dir: Path) -> Dict[str, str]:
    """在 data/gemma4_duplex_route1/audios 下生成真实 16kHz 0.48s 语音与旁人抗噪切片"""
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    clips: Dict[str, str] = {}

    # 复用仓库自带的真实女声咳嗽音频 mllm_demo_data/2.wav (截取 0.48s 16kHz)
    cough_src = REPO_ROOT / "data" / "mllm_demo_data" / "2.wav"
    if cough_src.exists():
        cough_dst = audio_out_dir / "cough_048s.wav"
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(cough_src), str(cough_dst)],
            check=False,
        )
        if cough_dst.exists():
            clips["noise"] = str(cough_dst.relative_to(REPO_ROOT / "data"))

    # 若在 macOS 上，额外合成几条真实中文 16kHz 语音分片样本放入训练集
    phrases = [
        ("u1_a.wav", "Tingting", "你好，帮我查一下"),
        ("u1_b.wav", "Tingting", "明天上海的天气"),
        ("barge.wav", "Tingting", "等一下，先别说了"),
    ]
    for fname, voice, txt in phrases:
        dst = audio_out_dir / fname
        aiff = audio_out_dir / (fname + ".aiff")
        if subprocess.run(["say", "-v", voice, "-o", str(aiff), txt], check=False).returncode == 0:
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(dst)],
                check=False,
            )
            aiff.unlink(missing_ok=True)
            if dst.exists():
                clips[fname] = str(dst.relative_to(REPO_ROOT / "data"))
    return clips


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pilot-jsonl",
        type=str,
        default=str(REPO_ROOT / "data" / "real_conversion_pilot_v1" / "train.jsonl"),
    )
    p.add_argument(
        "--tool-json",
        type=str,
        default=str(REPO_ROOT / "data" / "glaive_toolcall_zh_demo.json"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "gemma4_duplex_route1"),
    )
    args = p.parse_args()

    random.seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = generate_real_audio_clips(out_dir / "audios")
    noise_rel = clips.get("noise")

    all_entries: List[Dict[str, Any]] = []

    # 1. 转换本仓库 real_conversion_pilot_v1 数据集
    pilot_path = Path(args.pilot_jsonl)
    if pilot_path.exists():
        with open(pilot_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = parse_gemma4_raw_row(json.loads(line))
                if parsed:
                    sys_t, pairs = parsed
                    entry = build_chat_duplex_entry(sys_t, pairs, noise_wav_rel=noise_rel)
                    if entry:
                        entry.setdefault("audios", [])
                        all_entries.append(entry)

    # 2. 转换工具调用数据集 (glaive_toolcall_zh_demo.json)
    tool_path = Path(args.tool_json)
    if tool_path.exists():
        raw_tools = json.loads(tool_path.read_text(encoding="utf-8"))
        for item in raw_tools[:120]:
            all_entries.extend(build_tool_duplex_entries(item))

    # 3. 加入纯原生音频分片直入工具调用样本 (<audio> -> <|tool_call>)
    if "u1_a.wav" in clips and "u1_b.wav" in clips and "barge.wav" in clips:
        weather_tool_schema = json.dumps(
            [
                {
                    "name": "get_weather",
                    "description": "查询指定城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "城市名"}},
                        "required": ["city"],
                    },
                }
            ],
            ensure_ascii=False,
        )
        audio_tool_entry = {
            "conversations": [
                {"from": "human", "value": "<audio>"},
                {"from": "gpt", "value": SLOTS["user_talking"]},
                {"from": "human", "value": SLOTS["no_voice"]},
                {"from": "gpt", "value": SLOTS["user_talking"]},
                {"from": "human", "value": "<audio>"},
                {"from": "gpt", "value": SLOTS["user_talking"]},
                {"from": "human", "value": SLOTS["no_voice"]},
                {
                    "from": "gpt",
                    "value": f"{SLOTS['start_speak']}好的，马上查。{format_gemma4_tool_call('c1', 'get_weather', {'city': '上海'})}",
                },
                {"from": "human", "value": SLOTS["no_voice"]},
                {"from": "gpt", "value": SLOTS["waiting_tool"]},
                {"from": "observation", "value": '{"weather": "多云", "temp": "23度"}'},
                {"from": "gpt", "value": f"{SLOTS['start_speak']}明天上海多云，"},
                {"from": "human", "value": "<audio>"},
                {"from": "gpt", "value": SLOTS["interrupted"]},
            ],
            "tools": weather_tool_schema,
            "audios": [clips["u1_a.wav"], clips["u1_b.wav"], clips["barge.wav"]],
        }
        all_entries.extend([audio_tool_entry] * 6)

    random.shuffle(all_entries)
    out_file = out_dir / "train.json"
    out_file.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # 自动注册到 data/dataset_info.json
    ds_info_path = REPO_ROOT / "data" / "dataset_info.json"
    ds_info = json.loads(ds_info_path.read_text(encoding="utf-8"))
    ds_info["gemma4_duplex_route1"] = {
        "file_name": "gemma4_duplex_route1/train.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools",
            "audios": "audios",
        },
    }
    ds_info_path.write_text(json.dumps(ds_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audio_cnt = sum(1 for x in all_entries if x.get("audios"))
    tool_cnt = sum(1 for x in all_entries if x.get("tools") and x["tools"] != "[]")
    print(f"[完成] 共生成 {len(all_entries)} 条路线一全双工样本 (含工具样本 {tool_cnt} 条, 含真实音频块样本 {audio_cnt} 条)")
    print(f"[输出] {out_file}")
    print("[注册] 已写入 data/dataset_info.json -> dataset: gemma4_duplex_route1")


if __name__ == "__main__":
    main()
