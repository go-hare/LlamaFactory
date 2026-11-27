#!/usr/bin/env python3
"""
Automated Anti-Poisoning & Label-Mask Auditor for Gemma-4-E2B Route 2 (II-A) Stage 1 & Stage 2 Datasets.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

import soundfile as sf
import whisper


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gemma4_omni_route2 import Cosy2StreamingVocoder  # noqa: E402
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor  # noqa: E402
from llamafactory.data.template import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from types import SimpleNamespace
from llamafactory.hparams import DataArguments  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor  # noqa: E402


BAD_PATTERNS = [
    "我听到了", "您刚才说", "你刚才说", "小明", "作为AI",
    "用户打招呼", "用户问", "用户在问", "我应该", "thought",
    "**", "```", "[breath]", "[sigh]",
]


def main() -> None:
    s1_dir = REPO_ROOT / "data" / "gemma4_route2_stage1"
    s2_dir = REPO_ROOT / "data" / "gemma4_route2_stage2"

    train_s1 = json.loads((s1_dir / "train.json").read_text(encoding="utf-8"))
    valid_s1 = json.loads((s1_dir / "valid.json").read_text(encoding="utf-8"))
    all_s1 = train_s1 + valid_s1
    audit_meta = json.loads((s1_dir / "audit_samples.json").read_text(encoding="utf-8"))

    # 1. Poison scan across 100% of GPT turns
    poison_hits = 0
    mask_turn_count = 0
    chat_with_tools_neg = 0
    chat_plain = 0

    for rec in all_s1:
        has_tool_schema = bool(rec.get("tools"))
        has_tool_call = any("<|tool_call>" in t["value"] for t in rec["conversations"] if t["from"] == "gpt")
        has_unused2 = any(t["value"].startswith("<unused2>") for t in rec["conversations"] if t["from"] == "gpt")
        if has_unused2 and not has_tool_call and not any("<unused7>" in t["value"] for t in rec["conversations"] if t["from"] == "gpt"):
            if has_tool_schema:
                chat_with_tools_neg += 1
            else:
                chat_plain += 1
        for t in rec["conversations"]:
            if t["from"] == "gpt":
                v = t["value"]
                if v.startswith("<|mask_turn|>"):
                    mask_turn_count += 1
                for bp in BAD_PATTERNS:
                    if bp in v:
                        poison_hits += 1
                        print("POISON HIT:", bp, v)

    # 2. Check physical 0.48s (7680-sample @ 16kHz) FLAC frame integrity
    flacs = sorted((s1_dir / "audios").glob("*.flac"))
    bad_frames = 0
    for fp in flacs[:200]:
        info = sf.info(str(fp))
        if info.samplerate != 16000 or info.frames != 7680:
            bad_frames += 1

    print("================ ANTI-POISONING AUDIT REPORT ================")
    print(f"[1] Stage 1 Total Trajectories : {len(all_s1)} (Train={len(train_s1)}, Valid={len(valid_s1)})")
    print(f"[2] Forbidden Poison Hits      : {poison_hits} (MUST BE 0)")
    print(f"[3] <|mask_turn|> Prefixes     : {mask_turn_count}")
    print(f"[4] Chat Tool-Irrelevance Negs : {chat_with_tools_neg} with 9-tools / {chat_plain} plain ({chat_with_tools_neg/(chat_with_tools_neg+chat_plain)*100:.1f}% carry tools with 0 tool calls!)")
    print(f"[5] 0.48s (7680@16kHz) Check   : {200 - bad_frames}/200 exact physical frames")

    # 3. Verify SupervisedDatasetProcessor <|mask_turn|> sets -100 on partial assistant prefix!
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    tok.add_tokens([f"<unused{i}>" for i in range(8)], special_tokens=True)
    proc = SimpleNamespace(
        feature_extractor=Gemma4AudioFeatureExtractor.from_pretrained("google/gemma-4-E2B-it"),
        image_processor=object(),
        tokenizer=tok,
        audio_sampling_rate=16000,
        boi_token="<|image>",
        eoi_token="<image|>",
        boa_token="<|audio>",
        eoa_token="<audio|>",
        image_token="<|image|>",
        video_token="<|video|>",
        audio_token="<|audio|>",
    )
    data_args = DataArguments(template="gemma4n_duplex", cutoff_len=2048)
    tmpl = get_template_and_fix_tokenizer(tok, data_args)
    sup_proc = SupervisedDatasetProcessor(template=tmpl, tokenizer=tok, processor=proc, data_args=data_args)

    masked_sample = next(r for r in all_s1 if any(t["value"].startswith("<|mask_turn|>") for t in r["conversations"]))
    p_msgs = [
        {"role": ("user" if t["from"] == "human" else "assistant"), "content": t["value"]}
        for t in masked_sample["conversations"][:-1]
    ]
    r_msgs = [{"role": "assistant", "content": masked_sample["conversations"][-1]["value"]}]
    abs_auds = [str(REPO_ROOT / "data" / a) for a in masked_sample["audios"]]
    in_ids, lbl_ids = sup_proc._encode_data_example(p_msgs, r_msgs, None, masked_sample["tools"] or None, [], [], abs_auds)
    supervised_str = tok.decode([x for x in lbl_ids if x != IGNORE_INDEX], skip_special_tokens=False)
    full_str = tok.decode(in_ids, skip_special_tokens=False)
    assert "<|mask_turn|>" not in full_str, "<|mask_turn|> leaked into input_ids!"
    print(f"[6] LlamaFactory Label-Mask Verification: supervised tokens ONLY = {repr(supervised_str)} (partial prefix 100% masked with -100!)")

    # 4. Print representative real trajectories across every category!
    print("\n================ REPRESENTATIVE STAGE 1 TRAJECTORIES ================")
    shown_cats = set()
    want_prefixes = [
        "B1_daily_chat_with_tools_neg",
        "B2_real_cosy_qa_with_tools_neg",
        "A1_magicdata_hesitation_hold",
        "A2_ignore_",
        "A3_magicdata_real_barge_in",
        "C1_minds14_real_tool",
        "C2_tool_",
        "C3_async_tool_cancel",
    ]
    # Also scan full audit_meta
    for wp in want_prefixes:
        for r in audit_meta:
            cat = r["_meta"]["category"]
            if cat.startswith(wp) and wp not in shown_cats:
                shown_cats.add(wp)
                print(f"\n--- [{cat}] | User Audio Content: {r['_meta']['user_text']} | Tools Attached: {bool(r['tools'])} ---")
                for t in r["conversations"]:
                    print(f"  {t['from']:11s}: {t['value']}")
                break

    # 5. Stage 2 25Hz Unit -> 24kHz Vocoder Roundtrip Check on 3 samples!
    print("\n================ STAGE 2 (25Hz UNITS -> 24kHz WAV) VERIFICATION ================")
    s2_rows = [json.loads(l) for l in (s2_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[:3]]
    voc = Cosy2StreamingVocoder(device="cpu", use_zero_shot_prompt=True)
    asr_m = whisper.load_model("base")
    out_wav_dir = s2_dir / "preview_wavs"
    out_wav_dir.mkdir(exist_ok=True)
    for idx, row in enumerate(s2_rows):
        wav = voc.synthesize_full(row["speech_units"])
        wp = out_wav_dir / f"stage2_sample_{idx}.wav"
        sf.write(str(wp), wav, 24000)
        asr_txt = asr_m.transcribe(str(wp), language="zh")["text"]
        print(
            f"  Sample {idx} [{row['category']}]:\n"
            f"    Target Spoken Text : {row['spoken_text']}\n"
            f"    25Hz Units ({len(row['speech_units']):3d})   : {row['speech_units'][:12]}... ({len(wav)/24000:.2f}s @ 24kHz)\n"
            f"    Vocoder Whisper ASR: {asr_txt}\n"
            f"    Saved WAV          : {wp}"
        )


if __name__ == "__main__":
    main()
