#!/usr/bin/env python3
"""
Gemma-4-E2B Route-1 v3 生产级声学扩容 + 跨领域零 ASR 调工具 + 自回声抗干扰构建流水线

针对 v2 OOD 盲测成绩单（声学状态机 90~100%，唯独零 ASR 选工具 75%）做五大精准攻坚：
1. [能量谷智能切分] 淘汰时间硬切，在目标切点 ±250ms 内寻找短时 RMS 能量谷底切分，绝不劈断中文关键词。
2. [真实人类语音 2x 扩容] AISHELL-1 (shard 0+1, ~3,400 条真人) + CosyAudio (shard 0+1, ~4,000 条真实问答)
   + PolyAI/minds14 (zh-CN 4x 声学增强: 干净/AudioSet 叠噪/轻量 RIR/0.93~1.07x 语速微扰，严格保持 61 条 OOD 零泄漏)。
3. [跨领域零 ASR 音频调工具 (+2,000 条音频闭环)]
   - 抽取 900 条真实 hqfx/fc_zh_hard_oai 多域工具指令 + 100 条高频日常语音助手工具指令（天气/航班/导航/智能家居/闹钟/音乐/点餐/快递/汇率/翻译），
     用 14 个大陆/台湾多音色并行合成 16kHz 音频（严格 4 个独立音色留给 OOD 盲测！），专攻直接从声波抽中文实体参数 + 4 选 1 调工具！
4. [生产级声学边界补齐]
   - 自回声残留负样本 (<unused5>): 助手回复音频经 300~3400Hz 扬声器带通 + MIT RIR 房间混响 + -12~-20dB 衰减，防机器被自己扬声器声音打断！
   - 自然口语犹豫/卡顿停顿 (<unused1>): 多音色“呃——那个——”“嗯我想一下啊——”注入，专治句中抢话！
5. [10 维度严格零泄漏 OOD 基准] 产出 ood_eval.json（含未见 minds14 + 未见跨领域多音色调工具 + 未见自回声 + 未见犹豫卡顿 + 未见远场/噪音/嘈杂近场）。
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
import subprocess
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
PARQUET_CACHE = Path.home() / ".cache" / "gemma4_duplex_v2_parquets"
SR = 16000
CHUNK_SAMPLES = 7680  # 0.48s @ 16kHz = 12 Gemma-4 audio soft tokens

SLOTS = {
    "no_voice": "<unused0>",
    "user_talking": "<unused1>",
    "start_speak": "<unused2>",
    "waiting_tool": "<unused3>",
    "idle_think": "<unused4>",
    "ignore_noise": "<unused5>",
    "interrupted": "<unused6>",
    "cancel_tool": "<unused7>",
}

# 18 个 macOS 中文音色严格拆分为 14 Train / 4 OOD Held-Out (零说话人重叠!)
TRAIN_VOICES = [
    "Tingting",
    "Eddy (中文（中国大陆）)",
    "Flo (中文（中国大陆）)",
    "Grandma (中文（中国大陆）)",
    "Grandpa (中文（中国大陆）)",
    "Reed (中文（中国大陆）)",
    "Rocko (中文（中国大陆）)",
    "Sandy (中文（中国大陆）)",
    "Meijia",
    "Eddy (中文（台湾）)",
    "Flo (中文（台湾）)",
    "Grandma (中文（台湾）)",
    "Reed (中文（台湾）)",
    "Rocko (中文（台湾）)",
]
OOD_VOICES = [
    "Shelley (中文（中国大陆）)",
    "Sandy (中文（台湾）)",
    "Grandpa (中文（台湾）)",
    "Shelley (中文（台湾）)",
]

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
HESITATION_PHRASES_ZH = [
    "呃，那个",
    "嗯，我想一下啊",
    "就是刚才那个",
    "等一下，让我想想",
    "哦对了，那个",
    "嗯，我想问一下",
    "啊，怎么说来着",
    "就是昨天说的那个",
]

# 高频语音助手跨领域工具模板（用于构造零 ASR 中文实体参数抽取训练 + 独立 OOD 测试）
EVERYDAY_TOOL_DOMAINS: List[Dict[str, Any]] = [
    {
        "name": "get_weather",
        "desc": "查询指定城市和日期的天气预报",
        "params": {"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}, "required": ["city"]},
        "templates": [
            ("帮我查一下{date}{city}的天气怎么样", {"city": "{city}", "date": "{date}"}),
            ("{city}{date}会下雨吗帮我看看", {"city": "{city}", "date": "{date}"}),
            ("查查{city}{date}的气温和空气质量", {"city": "{city}", "date": "{date}"}),
        ],
        "slots": {
            "city": ["北京", "上海", "深圳", "杭州", "成都", "广州", "武汉", "南京", "西安", "苏州", "重庆", "青岛", "长沙", "厦门"],
            "date": ["今天", "明天", "后天", "这周末", "下周一"],
        },
        "obs": '{"weather":"多云转晴","temp":"18到25度","aqi":42}',
        "reply": "查到了，{city}{date}多云转晴，气温十八到二十五度，空气质量很不错。",
    },
    {
        "name": "search_flight",
        "desc": "查询两座城市之间的航班与票价信息",
        "params": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination"]},
        "templates": [
            ("帮我查一下{date}从{origin}飞{destination}的航班", {"origin": "{origin}", "destination": "{destination}", "date": "{date}"}),
            ("看看{date}{origin}到{destination}还有没有机票", {"origin": "{origin}", "destination": "{destination}", "date": "{date}"}),
        ],
        "slots": {
            "origin": ["北京", "上海", "广州", "深圳", "成都", "杭州"],
            "destination": ["三亚", "西安", "昆明", "厦门", "重庆", "哈尔滨", "青岛"],
            "date": ["明天", "后天", "周五", "下周二"],
        },
        "obs": '{"flights":[{"no":"CA1501","time":"09:30","price":780}]}',
        "reply": "{date}从{origin}飞{destination}最早一班是早上九点半起飞，票价七百八十元。",
    },
    {
        "name": "control_smart_home",
        "desc": "控制家中空调、灯光、窗帘或加湿器等智能设备",
        "params": {"type": "object", "properties": {"room": {"type": "string"}, "device": {"type": "string"}, "action": {"type": "string"}}, "required": ["device", "action"]},
        "templates": [
            ("帮我把{room}的{device}{action}一下", {"room": "{room}", "device": "{device}", "action": "{action}"}),
            ("麻烦{action}{room}里的{device}", {"room": "{room}", "device": "{device}", "action": "{action}"}),
        ],
        "slots": {
            "room": ["客厅", "主卧", "书房", "厨房", "次卧", "阳台"],
            "device": ["空调", "主灯", "窗帘", "加湿器", "空气净化器", "扫地机器人"],
            "action": ["打开", "关闭", "调成睡眠模式"],
        },
        "obs": '{"ok":true}',
        "reply": "好的，已经为您把{room}的{device}{action}了。",
    },
    {
        "name": "set_alarm",
        "desc": "设置闹钟或日程语音提醒",
        "params": {"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}}, "required": ["time"]},
        "templates": [
            ("帮我定一个{time}的闹钟提醒我{label}", {"time": "{time}", "label": "{label}"}),
            ("设个{time}的提醒事项内容是{label}", {"time": "{time}", "label": "{label}"}),
        ],
        "slots": {
            "time": ["明天早上七点", "明天早上八点半", "下午三点", "晚上九点", "十分钟后", "半小时后"],
            "label": ["开周会", "吃药", "去机场", "交报告", "接孩子", "健身"],
        },
        "obs": '{"alarm_id":"A01","status":"set"}',
        "reply": "没问题，已经设好了{time}{label}的提醒。",
    },
    {
        "name": "navigate_to",
        "desc": "规划驾车或公交导航路线并估算耗时",
        "params": {"type": "object", "properties": {"destination": {"type": "string"}, "mode": {"type": "string"}}, "required": ["destination"]},
        "templates": [
            ("帮我导航去{destination}，我准备{mode}过去", {"destination": "{destination}", "mode": "{mode}"}),
            ("查一下{mode}到{destination}现在堵不堵车", {"destination": "{destination}", "mode": "{mode}"}),
        ],
        "slots": {
            "destination": ["首都机场", "虹桥火车站", "西湖景区", "中关村", "国贸大厦", "奥林匹克公园", "人民广场", "南山科技园"],
            "mode": ["开车", "坐地铁", "骑车", "打车"],
        },
        "obs": '{"distance":"14.2公里","duration":"28分钟","traffic":"畅通"}',
        "reply": "现在{mode}去{destination}全程十四公里，大概二十八分钟，路况比较畅通。",
    },
    {
        "name": "play_music",
        "desc": "按歌手、曲风或心情播放音乐",
        "params": {"type": "object", "properties": {"artist_or_mood": {"type": "string"}}, "required": ["artist_or_mood"]},
        "templates": [
            ("帮我放一首{artist_or_mood}的歌吧", {"artist_or_mood": "{artist_or_mood}"}),
            ("我想听点{artist_or_mood}音乐放松一下", {"artist_or_mood": "{artist_or_mood}"}),
        ],
        "slots": {
            "artist_or_mood": ["周杰伦", "陈奕迅", "林俊杰", "轻音乐", "爵士乐", "助眠白噪音", "华语经典老歌", "民谣"],
        },
        "obs": '{"playing":true,"track":"已开始播放"}',
        "reply": "好的，马上为您播放{artist_or_mood}相关曲目。",
    },
    {
        "name": "track_package",
        "desc": "查询快递物流最新配送进度",
        "params": {"type": "object", "properties": {"courier": {"type": "string"}}, "required": ["courier"]},
        "templates": [
            ("帮我查一下我的{courier}快递到哪儿了", {"courier": "{courier}"}),
            ("看看那个{courier}包裹今天能不能送到", {"courier": "{courier}"}),
        ],
        "slots": {
            "courier": ["顺丰", "京东", "中通", "圆通", "韵达", "邮政"],
        },
        "obs": '{"status":"派送中","eta":"今天下午三点前"}',
        "reply": "您的{courier}包裹正在派送中，预计今天下午三点前送达。",
    },
    {
        "name": "query_exchange_rate",
        "desc": "查询实时外汇牌价与汇率换算",
        "params": {"type": "object", "properties": {"currency": {"type": "string"}}, "required": ["currency"]},
        "templates": [
            ("查一下今天{currency}兑人民币的汇率是多少", {"currency": "{currency}"}),
            ("现在一百{currency}能换多少人民币", {"currency": "{currency}"}),
        ],
        "slots": {
            "currency": ["美元", "日元", "欧元", "英镑", "港币", "韩元", "新加坡元"],
        },
        "obs": '{"rate":"实时牌价已更新"}',
        "reply": "查到今天的{currency}汇率比较平稳，牌价已经为您同步到屏幕上。",
    },
    {
        "name": "book_restaurant",
        "desc": "预订附近餐厅包间或座位",
        "params": {"type": "object", "properties": {"cuisine": {"type": "string"}, "people": {"type": "integer"}}, "required": ["cuisine", "people"]},
        "templates": [
            ("帮我订一家{cuisine}餐厅，今晚{people}个人吃饭", {"cuisine": "{cuisine}", "people": "{people}"}),
            ("查查附近评价好的{cuisine}，我要订{people}人桌", {"cuisine": "{cuisine}", "people": "{people}"}),
        ],
        "slots": {
            "cuisine": ["川菜", "粤菜", "日料", "火锅", "湘菜", "西餐", "烤鸭"],
            "people": [2, 4, 6, 8],
        },
        "obs": '{"booked":true,"time":"18:30"}',
        "reply": "已经帮您订好今晚六点半的{people}人{cuisine}座位。",
    },
    {
        "name": "translate_text",
        "desc": "将口语内容翻译为指定目标语言",
        "params": {"type": "object", "properties": {"target_lang": {"type": "string"}, "phrase": {"type": "string"}}, "required": ["target_lang", "phrase"]},
        "templates": [
            ("帮我把{phrase}翻译成{target_lang}", {"target_lang": "{target_lang}", "phrase": "{phrase}"}),
            ("{phrase}用{target_lang}怎么说", {"target_lang": "{target_lang}", "phrase": "{phrase}"}),
        ],
        "slots": {
            "target_lang": ["英语", "日语", "法语", "韩语", "德语", "西班牙语"],
            "phrase": ["请问洗手间在哪里", "这个多少钱", "很高兴认识你", "请给我一杯冰水", "我想办理入住"],
        },
        "obs": '{"translated":"ok"}',
        "reply": "好的，{phrase}翻译成{target_lang}的说法已经为您准备好。",
    },
]

MINDS14_INTENT_TOOLS: Dict[int, Dict[str, Any]] = {
    0: {"name": "check_abroad_usage", "desc": "查询银行卡境外消费与跨境使用规则", "params": {"type": "object", "properties": {"service": {"type": "string"}}, "required": []}, "args": {"service": "abroad"}, "obs": '{"enabled":true,"fee_rate":"1.5%"}', "reply": "您的银行卡已开通境外消费，跨境手续费率为百分之一点五。"},
    1: {"name": "update_address", "desc": "修改银行账户预留联系地址", "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": []}, "args": {"action": "update"}, "obs": '{"status":"verification_sent"}', "reply": "好的，已经向您手机发送了地址修改验证链接，确认后即可生效。"},
    2: {"name": "report_app_error", "desc": "排查并提交手机银行 App 报错工单", "params": {"type": "object", "properties": {"issue_type": {"type": "string"}}, "required": []}, "args": {"issue_type": "app_error"}, "obs": '{"ticket":"T8842","server_status":"normal"}', "reply": "服务器目前正常，已为您登记排查工单，建议先重启应用试试。"},
    3: {"name": "check_atm_limit", "desc": "查询银行卡 ATM 每日取现额度", "params": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": []}, "args": {"channel": "atm"}, "obs": '{"daily_limit":"20000元","remaining":"20000元"}', "reply": "您这张卡的取现日限额是两万元，今日可用额度还剩两万元。"},
    4: {"name": "check_balance", "desc": "查询银行账户当前可用余额", "params": {"type": "object", "properties": {"account_type": {"type": "string"}}, "required": []}, "args": {"account_type": "savings"}, "obs": '{"balance":"18650.50元","currency":"CNY"}', "reply": "您的账户当前可用余额是一万八千六百五十元五角。"},
    5: {"name": "query_business_loan", "desc": "查询企业与小微经营贷款申请条件", "params": {"type": "object", "properties": {"loan_type": {"type": "string"}}, "required": []}, "args": {"loan_type": "business"}, "obs": '{"max_amount":"50万元","apr":"3.85%"}', "reply": "目前小微经营贷最高可申请五十万元，年化利率三点八五起。"},
    6: {"name": "report_card_issue", "desc": "查询并处理银行卡异常、损坏或补换卡问题", "params": {"type": "object", "properties": {"issue": {"type": "string"}}, "required": []}, "args": {"issue": "card_problem"}, "obs": '{"card_status":"active","can_replace":true}', "reply": "查到您的卡片状态正常，如需补换新卡我可以马上为您登记。"},
    7: {"name": "query_cash_deposit", "desc": "查询现金存款渠道与单笔限额", "params": {"type": "object", "properties": {"method": {"type": "string"}}, "required": []}, "args": {"method": "cash"}, "obs": '{"atm_deposit_limit":"50000元"}', "reply": "自助存取款机单日最多可存五万元，超过金额需携带身份证去柜台。"},
    8: {"name": "manage_direct_debit", "desc": "查询或管理自动代扣与定期扣款协议", "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": []}, "args": {"action": "list"}, "obs": '{"active_debits":2,"next_date":"每月15日"}', "reply": "您名下目前有两项生效中的自动代扣，下一次扣款在每月十五号。"},
    9: {"name": "freeze_card", "desc": "紧急冻结或解冻银行卡", "params": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}, "args": {"action": "freeze"}, "obs": '{"card_frozen":true,"effective":"immediate"}', "reply": "已为您紧急冻结该银行卡，所有线上线下消费通道都已暂停。"},
    10: {"name": "query_high_value_payment", "desc": "查询大额转账限额与安全验证流程", "params": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": []}, "args": {"channel": "transfer"}, "obs": '{"single_limit":"50万元"}', "reply": "单笔大额转账最高支持五十万元，需要通过人脸或安全盾验证。"},
    11: {"name": "query_joint_account", "desc": "查询联名共同账户开立与管理规则", "params": {"type": "object", "properties": {"account_type": {"type": "string"}}, "required": []}, "args": {"account_type": "joint"}, "obs": '{"supported":true}', "reply": "联名账户需要两位持卡人携带本人身份证一同办理开户。"},
    12: {"name": "query_latest_transactions", "desc": "查询银行账户最近消费与入账流水", "params": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}, "args": {"limit": 3}, "obs": '{"latest":[{"merchant":"超市","amount":"-68.5元"}]}', "reply": "您最近两笔消费是超市六十八块五，还有咖啡店二十八元。"},
    13: {"name": "pay_bill", "desc": "办理账单缴费与信用卡还款", "params": {"type": "object", "properties": {"bill_type": {"type": "string"}}, "required": []}, "args": {"bill_type": "utility_or_credit"}, "obs": '{"due_amount":"326.00元","due_date":"本月28日"}', "reply": "查到您本期应缴账单是三百二十六元，截止日期是本月二十八号。"},
}

PARQUET_URLS = {
    "minds14_zh": "https://huggingface.co/datasets/PolyAI/minds14/resolve/refs%2Fconvert%2Fparquet/zh-CN/train/0000.parquet",
    "aishell1_0": "https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    "aishell1_1": "https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test/resolve/refs%2Fconvert%2Fparquet/default/test/0001.parquet",
    "cosy_qa_0": "https://huggingface.co/datasets/JerryAGENDD/chinese_speech_cosy_audio/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "cosy_qa_1": "https://huggingface.co/datasets/JerryAGENDD/chinese_speech_cosy_audio/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
}

ECHO_SOS = butter(4, [300.0, 3400.0], btype="bandpass", fs=SR, output="sos")


def download_parquet(name: str, url: str) -> Path:
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    dst = PARQUET_CACHE / f"{name}.parquet"
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        try:
            pq.read_metadata(dst)
            print(f"  [Cache HIT] {dst.name} ({dst.stat().st_size / 1024**2:.1f} MB)")
            return dst
        except Exception:
            print(f"  [Cache CORRUPT -> re-download] {dst.name}")
            dst.unlink(missing_ok=True)
    print(f"  [Download] {name} <- {url}")
    tmp = dst.with_suffix(".parquet.part")
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-sS", "-o", str(tmp), url],
        check=True,
    )
    pq.read_metadata(tmp)
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


def normalize_rms(x: np.ndarray, target_rms: float = 0.085) -> np.ndarray:
    r = rms(x)
    if r < 1e-6:
        return x
    return np.clip(x * (target_rms / r), -0.98, 0.98).astype(np.float32)


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """轻度语速/音高微扰 (0.93 ~ 1.07) 增强跨人声泛化"""
    up = int(round(100 / factor))
    down = 100
    g = math.gcd(up, down)
    return resample_poly(wav, up // g, down // g).astype(np.float32)


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


def make_self_echo_residual(wav: np.ndarray, rir: Optional[np.ndarray]) -> np.ndarray:
    """模拟真实全双工设备中 AEC 未消干净的助手扬声器自回声残留（300-3400Hz 带通 + 房间混响 + 非线性轻微削波 + 衰减）"""
    x = sosfilt(ECHO_SOS, wav).astype(np.float32)
    if rir is not None:
        x = fftconvolve(x, rir, mode="full")[: len(wav)].astype(np.float32)
    # 模拟 AEC 残余能量 (-12 ~ -20 dB) 与轻微扬声器失真
    x = np.tanh(x * 1.4).astype(np.float32)
    att_db = random.uniform(12.0, 20.0)
    return normalize_rms(x, target_rms=0.09 * (10.0 ** (-att_db / 20.0)))


def split_audio_energy_valley(wav: np.ndarray, n_parts: int) -> List[np.ndarray]:
    """
    能量谷底智能切分（v3 核心改进）：
    在目标切点 ±250ms 窗口内按 20ms 步长找短时能量最低的静音/词间停顿处切分，杜绝硬切劈断中文关键词！
    """
    wav = trim_silence(wav)
    if len(wav) < CHUNK_SAMPLES * 2 or n_parts <= 1:
        if len(wav) < CHUNK_SAMPLES:
            wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        return [wav]

    hop = 320  # 20ms
    win = 640  # 40ms
    n_frames = max(1, (len(wav) - win) // hop)
    energies = np.array([rms(wav[i * hop : i * hop + win]) for i in range(n_frames)], dtype=np.float32)

    cuts: List[int] = []
    min_gap = int(SR * 0.48)
    last_cut = 0
    for k in range(1, n_parts):
        ideal = int(len(wav) * k / n_parts)
        lo = max(last_cut + min_gap, ideal - int(SR * 0.28))
        hi = min(len(wav) - min_gap, ideal + int(SR * 0.28))
        if hi <= lo:
            continue
        f_lo = max(0, lo // hop)
        f_hi = min(n_frames - 1, hi // hop)
        if f_hi > f_lo:
            best_f = f_lo + int(np.argmin(energies[f_lo : f_hi + 1]))
            cut_pt = best_f * hop + win // 2
        else:
            cut_pt = ideal
        cuts.append(cut_pt)
        last_cut = cut_pt

    pts = [0] + sorted(cuts) + [len(wav)]
    parts = [wav[pts[i] : pts[i + 1]] for i in range(len(pts) - 1)]
    return [p for p in parts if len(p) >= int(SR * 0.35)]


def write_flac(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.clip(wav, -0.99, 0.99).astype(np.float32)
    sf.write(str(path), wav, SR, format="FLAC", subtype="PCM_16")


def synth_say_16k(text: str, voice: str, rate: int) -> Optional[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = tf.name
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "--file-format=WAVE", "--data-format=LEI16@16000", "-o", tmp_path, text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, _ = sf.read(tmp_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return trim_silence(wav)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
    if any(w in ans for w in ["哈哈", "开心", "太棒", "恭喜"]) and random.random() < 0.28:
        return "[laughter] " + ans
    if any(w in ans for w in ["难过", "遗憾", "抱歉", "辛苦", "累"]) and random.random() < 0.28:
        return "[sigh] " + ans
    if len(ans) > 22 and "，" in ans and random.random() < 0.20:
        parts = ans.split("，", 1)
        return f"{parts[0]}，[breath] {parts[1]}"
    return ans


def summarize_spoken_answer(text: str, max_len: int = 82) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[#*`>\-]", "", text)
    text = re.sub(r"\b\d+\.\s*", "", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
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


# =============================================================================
# 声学底池：MIT RIR (270) + AudioSet (450) + AISHELL-1 (shard 0+1, 3200 真人) + 自回声 + 犹豫卡顿
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
    sp = max(1, int(len(rirs) * 0.85))
    return rirs[:sp], rirs[sp:]


def load_audioset_bank(
    audio_dir: Path, data_rel_prefix: str, max_clips: int = 450
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    as_pqs = glob.glob(str(HF_CACHE / "datasets--agkphysics--AudioSet/snapshots/*/**/*.parquet"), recursive=True)
    items: List[Tuple[str, np.ndarray]] = []
    if as_pqs:
        tbl = pq.read_table(as_pqs[0])
        for i in range(min(tbl.num_rows, max_clips + 50)):
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
            clip = wav[random.randint(0, len(wav) - dur_n) :][:dur_n] if len(wav) > dur_n else wav
            if rms(clip) < 0.005:
                continue
            clip = normalize_rms(clip, target_rms=random.uniform(0.03, 0.09))
            sub = "tr_noise" if len(items) < int(max_clips * 0.9) else "ood_noise"
            rel = f"{data_rel_prefix}/audios/{sub}/as_{i:04d}.flac"
            write_flac(audio_dir / sub / f"as_{i:04d}.flac", clip)
            items.append((rel, clip))
    sp = int(len(items) * 0.9)
    return items[:sp], items[sp:]


def build_hesitation_and_echo_banks(
    audio_dir: Path,
    data_rel_prefix: str,
    rirs_tr: List[np.ndarray],
    rirs_ood: List[np.ndarray],
) -> Dict[str, List[str]]:
    """并行合成自然口语犹豫词库 (<unused1>) + 助手自回声残留库 (<unused5>)，严格 Train/OOD 说话人隔离"""
    echo_texts = [
        "查到了，明天北京晴转多云，气温十八到二十五度。",
        "您的账户当前可用余额是一万八千六百五十元。",
        "好的，已经为您把客厅的空调打开并调成二十六度。",
        "目前小微经营贷最高可申请五十万元，年化利率三点八五起。",
        "没问题，已经设好了明天早上七点半开周会的提醒。",
        "现在开车去首都机场全程十四公里，大概二十八分钟。",
        "您的顺丰包裹正在派送中，预计今天下午三点前送达。",
        "已经帮您订好今晚六点半的四人川菜包间。",
    ]

    jobs: List[Tuple[str, str, str, int, bool]] = []
    # 1) 犹豫卡顿词 (Train: 84, OOD: 20)
    for vi, v in enumerate(TRAIN_VOICES):
        for pi, ph in enumerate(HESITATION_PHRASES_ZH[:6]):
            jobs.append(("tr_hesit", f"hes_tr_{vi:02d}_{pi}.flac", v, random.randint(155, 200), False))
            jobs[-1] = (jobs[-1][0], jobs[-1][1], v, random.randint(155, 200), False, ph)  # type: ignore
    for vi, v in enumerate(OOD_VOICES):
        for pi, ph in enumerate(HESITATION_PHRASES_ZH[:5]):
            jobs.append(("ood_hesit", f"hes_ood_{vi:02d}_{pi}.flac", v, random.randint(160, 200), True, ph))  # type: ignore

    # 2) 自回声残留 (Train: 56, OOD: 16)
    for vi, v in enumerate(TRAIN_VOICES[:7]):
        for ti, txt in enumerate(echo_texts):
            jobs.append(("tr_echo", f"echo_tr_{vi:02d}_{ti}.flac", v, random.randint(185, 215), False, txt))  # type: ignore
    for vi, v in enumerate(OOD_VOICES):
        for ti, txt in enumerate(echo_texts[:4]):
            jobs.append(("ood_echo", f"echo_ood_{vi:02d}_{ti}.flac", v, random.randint(185, 215), True, txt))  # type: ignore

    res: Dict[str, List[str]] = {"tr_hesit": [], "ood_hesit": [], "tr_echo": [], "ood_echo": []}

    def _run(job: Tuple[Any, ...]) -> Optional[Tuple[str, str]]:
        bucket, fname, voice, rate, is_ood, text = job
        wav = synth_say_16k(text, voice, rate)
        if wav is None or len(wav) < 3200:
            return None
        if "echo" in bucket:
            seg = wav[: int(SR * 1.44)] if len(wav) > int(SR * 1.44) else wav
            rirs = rirs_ood if is_ood else rirs_tr
            wav = make_self_echo_residual(seg, random.choice(rirs) if rirs else None)
        else:
            wav = normalize_rms(wav, target_rms=random.uniform(0.075, 0.12))
            if len(wav) < CHUNK_SAMPLES:
                wav = np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
        write_flac(audio_dir / bucket / fname, wav)
        return bucket, f"{data_rel_prefix}/audios/{bucket}/{fname}"

    with ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(_run, jobs):
            if r:
                res[r[0]].append(r[1])
    return res


def build_aishell_v3_banks(
    aishell_pqs: List[Path],
    audio_dir: Path,
    data_rel_prefix: str,
    rirs_tr: List[np.ndarray],
    rirs_ood: List[np.ndarray],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    max_per_shard: int = 1600,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tr_farfield": [], "ood_farfield": [],
        "tr_barge": [], "ood_barge": [],
        "tr_backchannel": [], "ood_backchannel": [],
        "tr_stream": [], "ood_stream": [],
        "tr_echo_real": [], "ood_echo_real": [],
    }
    global_idx = 0
    for si, pq_path in enumerate(aishell_pqs):
        tbl = pq.read_table(pq_path)
        idxs = list(range(tbl.num_rows))
        random.shuffle(idxs)
        idxs = idxs[:max_per_shard]
        sp = int(len(idxs) * 0.90)
        for is_ood, sub_idxs in [(False, idxs[:sp]), (True, idxs[sp:])]:
            tag = "ood" if is_ood else "tr"
            rirs = rirs_ood if is_ood else rirs_tr
            noises = noises_ood if is_ood else noises_tr
            for row_i in sub_idxs:
                global_idx += 1
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
                uid = f"s{si}_{row_i:05d}"

                if (not is_ood and len(out["tr_stream"]) < 1000) or (is_ood and len(out["ood_stream"]) < 60):
                    parts = split_audio_energy_valley(wav_clean, n_parts=random.choice([2, 3]))
                    if len(parts) >= 2:
                        rels = []
                        for pi, p_wav in enumerate(parts):
                            bucket = f"{tag}_ais_{global_idx % 8}"
                            fname = f"ais_{uid}_p{pi}.flac"
                            write_flac(audio_dir / bucket / fname, p_wav)
                            rels.append(f"{data_rel_prefix}/audios/{bucket}/{fname}")
                        out[f"{tag}_stream"].append({"chunks": rels, "text": ans})

                if (not is_ood and len(out["tr_farfield"]) < 800) or (is_ood and len(out["ood_farfield"]) < 60):
                    seg_len = random.choice([CHUNK_SAMPLES, int(CHUNK_SAMPLES * 1.5), CHUNK_SAMPLES * 2])
                    seg = wav_clean[:seg_len] if len(wav_clean) >= seg_len else np.pad(wav_clean, (0, seg_len - len(wav_clean)))
                    far = fftconvolve(seg, random.choice(rirs), mode="full")[: len(seg)].astype(np.float32) if rirs else seg.copy()
                    far = normalize_rms(far, target_rms=0.10 * (10.0 ** (-random.uniform(14.0, 22.0) / 20.0)))
                    if noises and random.random() < 0.65:
                        far = mix_at_snr(far, random.choice(noises)[1], snr_db=random.uniform(-2.0, 6.0))
                    bucket = f"{tag}_farfield"
                    fname = f"far_{uid}.flac"
                    write_flac(audio_dir / bucket / fname, far)
                    out[f"{tag}_farfield"].append(f"{data_rel_prefix}/audios/{bucket}/{fname}")

                if (not is_ood and len(out["tr_barge"]) < 800) or (is_ood and len(out["ood_barge"]) < 60):
                    seg_len = min(len(wav_clean), random.choice([int(SR * 0.96), int(SR * 1.44), int(SR * 1.92)]))
                    barge = wav_clean[:seg_len]
                    if noises and random.random() < 0.58:
                        barge = mix_at_snr(barge, random.choice(noises)[1], snr_db=random.uniform(6.5, 15.0))
                    bucket = f"{tag}_barge"
                    fname = f"barge_{uid}.flac"
                    write_flac(audio_dir / bucket / fname, barge)
                    out[f"{tag}_barge"].append({"path": f"{data_rel_prefix}/audios/{bucket}/{fname}", "text": ans})

                if (not is_ood and len(out["tr_echo_real"]) < 300) or (is_ood and len(out["ood_echo_real"]) < 30):
                    seg_len = min(len(wav_clean), int(SR * 1.2))
                    ec = make_self_echo_residual(wav_clean[:seg_len], random.choice(rirs) if rirs else None)
                    bucket = f"{tag}_echo_real"
                    fname = f"ecr_{uid}.flac"
                    write_flac(audio_dir / bucket / fname, ec)
                    out[f"{tag}_echo_real"].append(f"{data_rel_prefix}/audios/{bucket}/{fname}")

                if (not is_ood and len(out["tr_backchannel"]) < 240) or (is_ood and len(out["ood_backchannel"]) < 25):
                    bc = wav_clean[:CHUNK_SAMPLES] if len(wav_clean) >= CHUNK_SAMPLES else np.pad(wav_clean, (0, CHUNK_SAMPLES - len(wav_clean)))
                    bc = normalize_rms(bc, target_rms=random.uniform(0.015, 0.032))
                    bucket = f"{tag}_backchannel"
                    fname = f"bc_{uid}.flac"
                    write_flac(audio_dir / bucket / fname, bc)
                    out[f"{tag}_backchannel"].append(f"{data_rel_prefix}/audios/{bucket}/{fname}")
    return out


# =============================================================================
# 模块 A：PolyAI/minds14 (zh-CN) 4x 声学增强 + 能量谷切分 (441 -> 1,764 Train, 61 零泄漏 OOD)
# =============================================================================
def build_minds14_v3_samples(
    minds_pq: Path,
    audio_dir: Path,
    data_rel_prefix: str,
    bank: Dict[str, Any],
    extra_bank: Dict[str, List[str]],
    rirs_tr: List[np.ndarray],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    tbl = pq.read_table(minds_pq)
    idxs = list(range(tbl.num_rows))
    random.shuffle(idxs)
    sp = int(len(idxs) * 0.88)
    tr_idxs, ood_idxs = idxs[:sp], idxs[sp:]

    tr_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    ood_cases: List[Dict[str, Any]] = []

    def _one(row_i: int, is_ood: bool, var: int) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
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
        echo_pool = (extra_bank["ood_echo"] + bank["ood_echo_real"]) if is_ood else (extra_bank["tr_echo"] + bank["tr_echo_real"])
        hesit_pool = extra_bank["ood_hesit"] if is_ood else extra_bank["tr_hesit"]
        noise_paths = [p for p, _ in noises]

        if not is_ood:
            if var == 1 and noises:
                wav = mix_at_snr(wav, random.choice(noises)[1], snr_db=random.uniform(10.0, 18.0))
            elif var == 2:
                wav = speed_perturb(wav, random.choice([0.93, 0.96, 1.04, 1.07]))
            elif var == 3 and rirs_tr:
                wav_r = fftconvolve(wav, random.choice(rirs_tr), mode="full")[: len(wav)].astype(np.float32)
                wav = normalize_rms(0.75 * wav + 0.25 * normalize_rms(wav_r, rms(wav)), rms(wav))

        # v3 能量谷底智能切分，绝不劈断中文关键词！
        parts = split_audio_energy_valley(wav, n_parts=random.choice([2, 3]) if var != 2 else 1)
        tag = "ood_minds" if is_ood else f"tr_minds_v{var}"
        part_paths: List[str] = []
        for pi, p_wav in enumerate(parts):
            fname = f"m14_{row_i:04d}_v{var}_p{pi}.flac"
            write_flac(audio_dir / tag / fname, p_wav)
            part_paths.append(f"{data_rel_prefix}/audios/{tag}/{fname}")

        spec = MINDS14_INTENT_TOOLS[intent]
        distractors = random.sample([k for k in MINDS14_INTENT_TOOLS if k != intent], k=3)
        tool_decls = [
            {"name": MINDS14_INTENT_TOOLS[k]["name"], "description": MINDS14_INTENT_TOOLS[k]["desc"], "parameters": MINDS14_INTENT_TOOLS[k]["params"]}
            for k in [intent] + distractors
        ]
        random.shuffle(tool_decls)
        tools_str = json.dumps(tool_decls, ensure_ascii=False)

        convs: List[Dict[str, str]] = []
        audios: List[str] = []

        # 20% 概率开头先来一段犹豫词音频 ("呃，那个") -> 必须 <unused1> 忍住不抢话！
        if hesit_pool and random.random() < 0.20:
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(random.choice(hesit_pool))
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        for pi, p_rel in enumerate(part_paths):
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(p_rel)
            convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
            if pi == 0 and len(part_paths) >= 2 and random.random() < 0.65:
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

        call_str = format_gemma4_tool_call("c1", spec["name"], spec["args"])
        convs.append({"from": "human", "value": SLOTS["no_voice"]})
        convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{random.choice(FILLERS_ZH)}{call_str}"})

        for _ in range(random.randint(1, 2)):
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["waiting_tool"]})

        if random.random() < 0.25 and barge_pool:
            convs.append({"from": "human", "value": "<audio>"})
            audios.append(random.choice(barge_pool)["path"])
            convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已为您取消刚才的操作。"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
        else:
            convs.append({"from": "observation", "value": spec["obs"]})
            reply_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(spec["reply"]), 4, 8)
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{reply_chunks[0]}"})
            for ci, rc in enumerate(reply_chunks[1:]):
                r = random.random()
                if ci == 0 and echo_pool and r < 0.28:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(echo_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{rc}"})
                elif ci == 0 and far_pool and r < 0.58:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(far_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{rc}"})
                elif ci == 0 and noise_paths and r < 0.80:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(noise_paths))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{rc}"})
                elif ci == len(reply_chunks) - 2 and barge_pool and random.random() < 0.35:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(barge_pool)["path"])
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
            "echo_path": random.choice(echo_pool) if echo_pool else None,
            "hesit_path": random.choice(hesit_pool) if hesit_pool else None,
            "barge_path": random.choice(barge_pool)["path"] if barge_pool else None,
        }
        return sample, ood_meta

    for row_i in tr_idxs:
        for var in (0, 1, 2, 3):
            res = _one(row_i, is_ood=False, var=var)
            if res:
                tr_rows.append(res[0])

    for row_i in ood_idxs:
        res = _one(row_i, is_ood=True, var=0)
        if res:
            val_rows.append(res[0])
            if len(ood_cases) < 25:
                ood_cases.append(res[1])

    return tr_rows, val_rows, ood_cases


# =============================================================================
# 模块 B：跨领域零 ASR 音频调工具 (+1,800 Train / 40 严格未见音色 OOD，专攻 75% -> 95%+)
# =============================================================================
def load_hqfx_base_items() -> List[Dict[str, Any]]:
    pqs = glob.glob(str(HF_CACHE / "datasets--hqfx--fc_zh_hard_oai/snapshots/*/**/*.parquet"), recursive=True)
    if not pqs:
        return []
    tbl = pq.read_table(pqs[0])
    items: List[Dict[str, Any]] = []
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
            if not (5 <= len(user_q) <= 75):
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
            obs_str = str(m_obs.get("content") or "{}").strip()[:260]
            final_ans = summarize_spoken_answer(str(m_ans.get("content") or ""), max_len=80)
            if len(final_ans) < 8:
                continue
            items.append({
                "tools": json.dumps(norm_tools[:5], ensure_ascii=False),
                "q": user_q,
                "name": call_name,
                "args": call_args,
                "obs": obs_str,
                "ans": final_ans,
            })
            break
    return items


def generate_everyday_tool_items(n_train: int = 500, n_ood: int = 40) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_decls = [{"name": d["name"], "description": d["desc"], "parameters": d["params"]} for d in EVERYDAY_TOOL_DOMAINS]

    def _make_one() -> Dict[str, Any]:
        dom = random.choice(EVERYDAY_TOOL_DOMAINS)
        vals = {k: random.choice(vs) for k, vs in dom["slots"].items()}
        tpl_q, tpl_args = random.choice(dom["templates"])
        q = tpl_q.format(**vals)
        args = {}
        for ak, av in tpl_args.items():
            if isinstance(av, str) and av.startswith("{") and av.endswith("}"):
                args[ak] = vals[av[1:-1]]
            else:
                args[ak] = av
        distractors = random.sample([d for d in all_decls if d["name"] != dom["name"]], k=3)
        me = {"name": dom["name"], "description": dom["desc"], "parameters": dom["params"]}
        tlist = [me] + distractors
        random.shuffle(tlist)
        return {
            "tools": json.dumps(tlist, ensure_ascii=False),
            "q": q,
            "name": dom["name"],
            "args": args,
            "obs": dom["obs"],
            "ans": dom["reply"].format(**vals),
        }

    tr = [_make_one() for _ in range(n_train)]
    ood = [_make_one() for _ in range(n_ood)]
    return tr, ood


def build_cross_domain_audio_tool_samples(
    hqfx_items: List[Dict[str, Any]],
    audio_dir: Path,
    data_rel_prefix: str,
    bank: Dict[str, Any],
    extra_bank: Dict[str, List[str]],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    n_hqfx_audio: int = 450,
    n_everyday_tr: int = 450,
    n_ood: int = 35,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    合成 900 条跨领域真实中文工具语音指令 (hqfx + 10 大日常助手域) x 2 声学变体 = 1,800 Train
    + 35 条严格未见说话人 OOD 跨领域音频调工具测试（验证实体抽参 + 4 选 1 泛化）！
    """
    ev_tr, ev_ood = generate_everyday_tool_items(n_train=n_everyday_tr, n_ood=n_ood)
    short_hqfx = [x for x in hqfx_items if 8 <= len(x["q"]) <= 38 and is_clean_spoken_zh(x["q"], 6, 40)]
    random.shuffle(short_hqfx)
    tr_spec_list = ev_tr + short_hqfx[:n_hqfx_audio]
    random.shuffle(tr_spec_list)

    tr_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    ood_cases: List[Dict[str, Any]] = []

    jobs: List[Tuple[int, bool, Dict[str, Any], str, int]] = []
    for idx, sp in enumerate(tr_spec_list):
        jobs.append((idx, False, sp, TRAIN_VOICES[idx % len(TRAIN_VOICES)], random.randint(165, 225)))
    for idx, sp in enumerate(ev_ood):
        jobs.append((idx, True, sp, OOD_VOICES[idx % len(OOD_VOICES)], random.randint(170, 220)))

    def _synth_job(job: Tuple[int, bool, Dict[str, Any], str, int]) -> Optional[Tuple[int, bool, Dict[str, Any], np.ndarray]]:
        idx, is_ood, sp, voice, rate = job
        wav = synth_say_16k(sp["q"], voice, rate)
        if wav is None or len(wav) < CHUNK_SAMPLES:
            return None
        return idx, is_ood, sp, normalize_rms(wav, target_rms=random.uniform(0.08, 0.13))

    synth_results: List[Tuple[int, bool, Dict[str, Any], np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(_synth_job, jobs):
            if r is not None:
                synth_results.append(r)

    for idx, is_ood, sp, base_wav in synth_results:
        noises = noises_ood if is_ood else noises_tr
        far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
        barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
        echo_pool = (extra_bank["ood_echo"] + bank["ood_echo_real"]) if is_ood else (extra_bank["tr_echo"] + bank["tr_echo_real"])
        hesit_pool = extra_bank["ood_hesit"] if is_ood else extra_bank["tr_hesit"]
        noise_paths = [p for p, _ in noises]

        variants = (0,) if is_ood else (0, 1)
        for var in variants:
            wav = base_wav.copy()
            if var == 1 and noises:
                wav = mix_at_snr(wav, random.choice(noises)[1], snr_db=random.uniform(9.0, 17.0))
            parts = split_audio_energy_valley(wav, n_parts=random.choice([2, 3]))
            bucket = "ood_xtool" if is_ood else f"tr_xtool_s{idx % 6}"
            part_paths: List[str] = []
            for pi, p_wav in enumerate(parts):
                fname = f"xt_{idx:04d}_v{var}_p{pi}.flac"
                write_flac(audio_dir / bucket / fname, p_wav)
                part_paths.append(f"{data_rel_prefix}/audios/{bucket}/{fname}")

            convs: List[Dict[str, str]] = []
            audios: List[str] = []
            if hesit_pool and random.random() < 0.25:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(hesit_pool))
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            for pi, p_rel in enumerate(part_paths):
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(p_rel)
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                if pi == 0 and len(part_paths) >= 2 and random.random() < 0.6:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            call_str = format_gemma4_tool_call("c1", sp["name"], sp["args"])
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{random.choice(FILLERS_ZH)}{call_str}"})

            for _ in range(random.randint(1, 2)):
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["waiting_tool"]})

            if random.random() < 0.22 and barge_pool:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(barge_pool)["path"])
                convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已经取消刚才的查询。"})
                convs.append({"from": "human", "value": SLOTS["no_voice"]})
                convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
            else:
                convs.append({"from": "observation", "value": sp["obs"]})
                a_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(sp["ans"]), 4, 8)
                convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{a_chunks[0]}"})
                for ci, ac in enumerate(a_chunks[1:]):
                    r = random.random()
                    if ci == 0 and echo_pool and r < 0.35:
                        convs.append({"from": "human", "value": "<audio>"})
                        audios.append(random.choice(echo_pool))
                        convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                    elif ci == 0 and far_pool and r < 0.65:
                        convs.append({"from": "human", "value": "<audio>"})
                        audios.append(random.choice(far_pool))
                        convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ac}"})
                    elif ci == len(a_chunks) - 2 and barge_pool and random.random() < 0.30:
                        convs.append({"from": "human", "value": "<audio>"})
                        audios.append(random.choice(barge_pool)["path"])
                        convs.append({"from": "gpt", "value": SLOTS["interrupted"]})
                        break
                    else:
                        convs.append({"from": "human", "value": SLOTS["no_voice"]})
                        convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{ac}"})
                else:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": SLOTS["idle_think"]})

            row = {"conversations": convs, "tools": sp["tools"], "audios": audios}
            if is_ood:
                val_rows.append(row)
                ood_cases.append({
                    "type": "cross_domain_zero_asr_tool",
                    "expected_tool": sp["name"],
                    "expected_args": sp["args"],
                    "transcript": sp["q"],
                    "tools": sp["tools"],
                    "part_paths": part_paths,
                    "obs": sp["obs"],
                    "echo_path": random.choice(echo_pool) if echo_pool else None,
                    "hesit_path": random.choice(hesit_pool) if hesit_pool else None,
                    "far_path": random.choice(far_pool) if far_pool else None,
                    "barge_path": random.choice(barge_pool)["path"] if barge_pool else None,
                })
            else:
                tr_rows.append(row)

    return tr_rows, val_rows, ood_cases


# =============================================================================
# 模块 C：CosyAudio (shard 0+1) 真实中文语音问答流 + 能量谷切分 + 自回声/远场/真打断
# =============================================================================
def build_cosy_qa_v3_samples(
    cosy_pqs: List[Path],
    audio_dir: Path,
    data_rel_prefix: str,
    bank: Dict[str, Any],
    extra_bank: Dict[str, List[str]],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    max_per_shard: int = 1900,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tr_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []

    for si, pq_path in enumerate(cosy_pqs):
        tbl = pq.read_table(pq_path)
        idxs = list(range(tbl.num_rows))
        random.shuffle(idxs)
        kept = 0
        for row_i in idxs:
            if kept >= max_per_shard:
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

            is_ood = kept >= int(max_per_shard * 0.93)
            tag = f"ood_cosy_s{si}" if is_ood else f"tr_cosy_s{si}_{kept % 6}"
            far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
            barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
            bc_pool = bank["ood_backchannel"] if is_ood else bank["tr_backchannel"]
            echo_pool = (extra_bank["ood_echo"] + bank["ood_echo_real"]) if is_ood else (extra_bank["tr_echo"] + bank["tr_echo_real"])
            hesit_pool = extra_bank["ood_hesit"] if is_ood else extra_bank["tr_hesit"]
            noises = noises_ood if is_ood else noises_tr
            noise_paths = [p for p, _ in noises]

            if noises and random.random() < 0.30:
                wav = mix_at_snr(wav, random.choice(noises)[1], snr_db=random.uniform(10.0, 18.0))

            parts = split_audio_energy_valley(wav, n_parts=random.choice([2, 3]))
            part_paths: List[str] = []
            for pi, p_wav in enumerate(parts):
                fname = f"cqa_s{si}_{row_i:05d}_p{pi}.flac"
                write_flac(audio_dir / tag / fname, p_wav)
                part_paths.append(f"{data_rel_prefix}/audios/{tag}/{fname}")

            convs: List[Dict[str, str]] = []
            audios: List[str] = []
            if hesit_pool and random.random() < 0.18:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(hesit_pool))
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            for pi, p_rel in enumerate(part_paths):
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(p_rel)
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                if pi < len(part_paths) - 1 and random.random() < 0.55:
                    convs.append({"from": "human", "value": SLOTS["no_voice"]})
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(ans), 4, 7)
            if not chunks:
                continue
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}{chunks[0]}"})

            interrupted = False
            for ci, ch in enumerate(chunks[1:]):
                r = random.random()
                if ci == 0 and echo_pool and r < 0.22:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(echo_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif ci == 0 and far_pool and r < 0.45:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(far_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif ci == 0 and noise_paths and r < 0.65:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(noise_paths))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif ci == 1 and bc_pool and r < 0.25:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(bc_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif ci >= 1 and barge_pool and random.random() < 0.22:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(barge_pool)["path"])
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
            (val_rows if is_ood else tr_rows).append(row)
            kept += 1
    return tr_rows, val_rows


def build_aishell_contrast_v3(
    bank: Dict[str, Any],
    extra_bank: Dict[str, List[str]],
    noises_tr: List[Tuple[str, np.ndarray]],
    noises_ood: List[Tuple[str, np.ndarray]],
    spoken_pairs: List[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def _make(is_ood: bool, count: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        stream_pool = bank["ood_stream"] if is_ood else bank["tr_stream"]
        far_pool = bank["ood_farfield"] if is_ood else bank["tr_farfield"]
        barge_pool = bank["ood_barge"] if is_ood else bank["tr_barge"]
        bc_pool = bank["ood_backchannel"] if is_ood else bank["tr_backchannel"]
        echo_pool = (extra_bank["ood_echo"] + bank["ood_echo_real"]) if is_ood else (extra_bank["tr_echo"] + bank["tr_echo_real"])
        hesit_pool = extra_bank["ood_hesit"] if is_ood else extra_bank["tr_hesit"]
        noise_paths = [p for p, _ in (noises_ood if is_ood else noises_tr)]
        if not (far_pool and barge_pool and noise_paths):
            return rows

        for i in range(count):
            convs: List[Dict[str, str]] = []
            audios: List[str] = []
            if hesit_pool and random.random() < 0.25:
                convs.append({"from": "human", "value": "<audio>"})
                audios.append(random.choice(hesit_pool))
                convs.append({"from": "gpt", "value": SLOTS["user_talking"]})

            if stream_pool and i % 2 == 0:
                st = stream_pool[i % len(stream_pool)]
                for ci, p_rel in enumerate(st["chunks"]):
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(p_rel)
                    convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
                    if ci == 0 and random.random() < 0.55:
                        convs.append({"from": "human", "value": SLOTS["no_voice"]})
                        convs.append({"from": "gpt", "value": SLOTS["user_talking"]})
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

            interrupted = False
            for ch in chunks[1:]:
                p = random.random()
                if p < 0.25 and echo_pool:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(echo_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.52:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(far_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.70:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(noise_paths))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.78 and bc_pool:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(bc_pool))
                    convs.append({"from": "gpt", "value": f"{SLOTS['ignore_noise']}{ch}"})
                elif p < 0.92:
                    convs.append({"from": "human", "value": "<audio>"})
                    audios.append(random.choice(barge_pool)["path"])
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

    return _make(False, 3600), _make(True, 220)


def load_spoken_dialogue_pairs(max_pairs: int = 14000) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
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


def build_spoken_text_rows(pairs: List[Tuple[str, str]], count: int = 8500) -> List[Dict[str, Any]]:
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
            a_chunks = split_speech_chunks(inject_cosyvoice_paralinguistics(a), 4, 7)
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


def build_async_tool_text_rows(hqfx_items: List[Dict[str, Any]], max_rows: int = 6500) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not hqfx_items:
        return rows
    for i in range(max_rows):
        it = hqfx_items[i % len(hqfx_items)]
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
        r = random.random()
        if r < 0.30:
            convs.append({"from": "human", "value": random.choice(CANCEL_UTTERANCES_ZH)})
            convs.append({"from": "gpt", "value": f"{SLOTS['interrupted']}{SLOTS['cancel_tool']}c1"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": f"{SLOTS['start_speak']}好的，已为您取消刚才的操作。"})
            convs.append({"from": "human", "value": SLOTS["no_voice"]})
            convs.append({"from": "gpt", "value": SLOTS["idle_think"]})
        elif r < 0.50:
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
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "data" / "gemma4_duplex_route1_v3"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audios"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_rel_prefix = out_dir.name

    print("🚀 [1/8] 下载/读取真实中文语音 Parquet 分片 (minds14 + aishell1 x2 + cosy_qa x2)...")
    minds_pq = download_parquet("minds14_zh", PARQUET_URLS["minds14_zh"])
    ais0_pq = download_parquet("aishell1_0", PARQUET_URLS["aishell1_0"])
    ais1_pq = download_parquet("aishell1_1", PARQUET_URLS["aishell1_1"])
    cosy0_pq = download_parquet("cosy_qa_0", PARQUET_URLS["cosy_qa_0"])
    cosy1_pq = download_parquet("cosy_qa_1", PARQUET_URLS["cosy_qa_1"])

    print("🔊 [2/8] 加载 MIT RIR (270) + AudioSet (450) + 合成犹豫卡顿(<unused1>) & 自回声残留(<unused5>)库...")
    rirs_tr, rirs_ood = load_mit_rirs()
    noises_tr, noises_ood = load_audioset_bank(audio_dir, data_rel_prefix, max_clips=450)
    extra_bank = build_hesitation_and_echo_banks(audio_dir, data_rel_prefix, rirs_tr, rirs_ood)
    print(f"   hesit(tr/ood)={len(extra_bank['tr_hesit'])}/{len(extra_bank['ood_hesit'])} | echo(tr/ood)={len(extra_bank['tr_echo'])}/{len(extra_bank['ood_echo'])}")

    print("🎙️ [3/8] 构建 AISHELL-1 (双分片 ~3,200 真人) 能量谷切片 + 远场/嘈杂近场/自回声库...")
    bank = build_aishell_v3_banks([ais0_pq, ais1_pq], audio_dir, data_rel_prefix, rirs_tr, rirs_ood, noises_tr, noises_ood, max_per_shard=1600)
    print(f"   AISHELL: tr_stream={len(bank['tr_stream'])}, tr_far={len(bank['tr_farfield'])}, tr_barge={len(bank['tr_barge'])}, tr_echo={len(bank['tr_echo_real'])}")

    print("🏦 [4/8] 构建 PolyAI/minds14 (zh-CN) 4x 声学增强 + 能量谷切分 (441 -> 1,764 Train, 61 OOD)...")
    m14_tr, m14_val, ood_m14 = build_minds14_v3_samples(minds_pq, audio_dir, data_rel_prefix, bank, extra_bank, rirs_tr, noises_tr, noises_ood)
    print(f"   minds14: train={len(m14_tr)}, valid={len(m14_val)}, ood={len(ood_m14)}")

    print("🛠️ [5/8] 合成跨领域多音色零 ASR 音频调工具库 (900 独立指令 x2 = 1,800 Train + 35 未见音色 OOD)...")
    hqfx_items = load_hqfx_base_items()
    xt_tr, xt_val, ood_xt = build_cross_domain_audio_tool_samples(
        hqfx_items, audio_dir, data_rel_prefix, bank, extra_bank, noises_tr, noises_ood,
        n_hqfx_audio=450, n_everyday_tr=450, n_ood=35,
    )
    print(f"   cross_domain_audio_tools: train={len(xt_tr)}, valid={len(xt_val)}, ood={len(ood_xt)}")

    print("💬 [6/8] 构建 CosyAudio (双分片 ~3,800 真实问答流) + AISHELL 声学对比样本 (3,600)...")
    spoken_pairs = load_spoken_dialogue_pairs(max_pairs=14000)
    cosy_tr, cosy_val = build_cosy_qa_v3_samples([cosy0_pq, cosy1_pq], audio_dir, data_rel_prefix, bank, extra_bank, noises_tr, noises_ood, max_per_shard=1900)
    ais_tr, ais_val = build_aishell_contrast_v3(bank, extra_bank, noises_tr, noises_ood, spoken_pairs)
    print(f"   cosy_qa: tr={len(cosy_tr)}, val={len(cosy_val)} | aishell_contrast: tr={len(ais_tr)}, val={len(ais_val)}")

    print("📝 [7/8] 构建口语多轮短对话 (8,500) + 强化异步工具文本 (6,500)...")
    text_sp = build_spoken_text_rows(spoken_pairs, count=8500)
    text_fc = build_async_tool_text_rows(hqfx_items, max_rows=6500)
    random.shuffle(text_sp)
    random.shuffle(text_fc)
    sp_s, fc_s = int(len(text_sp) * 0.96), int(len(text_fc) * 0.96)

    train_all = m14_tr + xt_tr + cosy_tr + ais_tr + text_sp[:sp_s] + text_fc[:fc_s]
    valid_all = m14_val + xt_val + cosy_val + ais_val + text_sp[sp_s:] + text_fc[fc_s:]
    random.shuffle(train_all)
    random.shuffle(valid_all)

    for sname, rows in [("train", train_all), ("valid", valid_all)]:
        for r in rows:
            n_tag = sum(m["value"].count("<audio>") for m in r["conversations"])
            assert n_tag == len(r["audios"]), f"{sname} mismatch: {n_tag} vs {len(r['audios'])}"

    ood_manifest = {
        "minds14_tool_cases": ood_m14,
        "cross_domain_tool_cases": ood_xt,
        "ood_farfield": bank["ood_farfield"],
        "ood_barge": bank["ood_barge"],
        "ood_noise": [p for p, _ in noises_ood],
        "ood_echo": extra_bank["ood_echo"] + bank["ood_echo_real"],
        "ood_hesit": extra_bank["ood_hesit"],
    }

    (out_dir / "train.json").write_text(json.dumps(train_all, ensure_ascii=False), encoding="utf-8")
    (out_dir / "valid.json").write_text(json.dumps(valid_all, ensure_ascii=False), encoding="utf-8")
    (out_dir / "ood_eval.json").write_text(json.dumps(ood_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("📦 [8/8] 打包无损 FLAC 音频库 -> audios.tar ...")
    tar_path = out_dir / "audios.tar"
    flacs = sorted(audio_dir.rglob("*.flac"))
    tot_sec = 0.0
    with tarfile.open(tar_path, "w") as tf:
        for f in flacs:
            tot_sec += sf.info(str(f)).duration
            tf.add(f, arcname=str(f.relative_to(out_dir)))

    n_aud_tr = sum(1 for r in train_all if r["audios"])
    n_aud_val = sum(1 for r in valid_all if r["audios"])
    print("=" * 74)
    print(f"✅ v3 生产级声学 + 跨领域音频调工具数据集构建完成: {out_dir}")
    print(f"   Train: {len(train_all):,} 条 (含真实/口语音频: {n_aud_tr:,} 条, {100*n_aud_tr/len(train_all):.1f}%)")
    print(f"   Valid: {len(valid_all):,} 条 (含零泄漏音频: {n_aud_val:,} 条)")
    print(f"   独立 FLAC 音频: {len(flacs):,} 段 | 总时长: {tot_sec/3600:.2f} 小时 | audios.tar: {tar_path.stat().st_size / 1024**2:.1f} MB")
    print("=" * 74)


if __name__ == "__main__":
    main()
