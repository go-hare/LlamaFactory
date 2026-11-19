# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# ruff: noqa: E402

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from test_diffusion_ple import tiny_model


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_length_seeds import main, score_result, summarize, validate_plan
from run_single_sample_controls import save_checkpoint, source_hash
from validate_diffusion_conversion import ByteTokenizer


def fixture_checkpoint(root):
    root.mkdir()
    from llamafactory.train.diffusion.loss import _get_conditioner

    base = tiny_model()
    config = base.config
    config.text_config.vocab_size = 259
    model = type(base)(config).to(torch.bfloat16)
    _get_conditioner(model)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(model, ByteTokenizer(), optimizer, root, {"global_step": 800})
    (root / "settings.json").write_text(
        json.dumps(
            {
                "suite": "length",
                "tiny": True,
                "seeds": [42, 43, 44, 45],
                "canvas": 16,
                "sample_steps": 2,
                "code_sha256": source_hash(),
            }
        )
    )
    rows = [
        {"kind": "short_text", "prompt": "short", "response": "abc"},
        {"kind": "cross_block_text", "prompt": "long", "response": "abcdefghijklmno0123456789"},
    ]
    (root / "tasks.json").write_text(json.dumps({"train": rows}))
    return root / "checkpoint-800"


def test_score_raw_token_offsets_and_missing_block():
    row = {"response": "answer"}
    target = torch.arange(20)
    output = target.clone()
    output[18] = 99
    result = score_result(row, 100, "bad", output, target, 16, 0.0)
    assert result["blocks"][0]["exact"]
    assert result["blocks"][1]["first_wrong_offsets"] == [2]
    assert result["blocks"][1]["wrong_count"] == 1
    truncated = score_result(row, 100, "bad", target[:3], target, 16, 0.0)
    assert truncated["blocks"][1]["wrong_count"] == 4


def test_plan_rejects_old_seeds_and_source_changes(tmp_path):
    ckpt = fixture_checkpoint(tmp_path / "experiment")
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(ckpt, [42, 100], True)
    with pytest.raises(ValueError, match="unique"):
        validate_plan(ckpt, [100, 100], True)
    p = ckpt.parent / "settings.json"
    settings = json.loads(p.read_text())
    settings["code_sha256"] = "bad"
    p.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError, match="hash differs"):
        validate_plan(ckpt, [100], True)


def test_audit_never_creates_optimizer_and_preserves_checkpoint(tmp_path, capsys):
    ckpt = fixture_checkpoint(tmp_path / "experiment")
    before = {
        str(p.relative_to(ckpt.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ckpt.parent.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "audit"
    argv = ["audit", "--tiny", "--checkpoint", str(ckpt), "--out", str(output), "--seeds", "100", "101"]
    with (
        patch.object(sys, "argv", argv),
        patch("torch.optim.AdamW", side_effect=AssertionError("Audit must not create optimizer")),
    ):
        main()
    after = {
        str(p.relative_to(ckpt.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ckpt.parent.rglob("*")
        if p.is_file()
    }
    assert before == after
    samples = [json.loads(s) for s in (output / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 4
    assert {(s["row_index"], s["seed"]) for s in samples} == {(0, 100), (0, 101), (1, 100), (1, 101)}
    summary = summarize(output / "samples.jsonl")
    assert summary["attempts"] == 4
    assert len(summary["tasks"]) == 2
    assert not (output / "diagnostic_state.pt").exists()
    with patch.object(sys, "argv", argv), pytest.raises(RuntimeError, match="nothing overwritten"):
        main()
    with (
        patch.object(sys, "argv", ["audit", "--out", str(output), "--summary-only"]),
        patch("torch.cuda.is_available", side_effect=AssertionError("Summary must not inspect CUDA")),
    ):
        main()
    assert "SEED_AUDIT_DONE" in capsys.readouterr().out


def test_cli_old_seed_fails_before_output_creation(tmp_path):
    ckpt = fixture_checkpoint(tmp_path / "experiment")
    out = tmp_path / "rejected"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_length_seeds.py"),
            "--tiny",
            "--checkpoint",
            str(ckpt),
            "--out",
            str(out),
            "--seeds",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "overlap" in result.stderr
    assert not out.exists()
