# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# ruff: noqa: E402

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_diffusion_ple import tiny_model


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_single_sample_controls import compare_blocks, evaluate, length_metadata, task_rows


def test_length_targets_and_block_boundaries():
    train, heldout = task_rows("length")
    assert len(train) == 2 and len(heldout) == 1
    assert train[0]["kind"] == "short_text" and train[1]["kind"] == "cross_block_text"

    class Tok:
        def encode(self, text, **kwargs):
            return list(range(49 if text.startswith("先把水杯") else 293))

    lengths = length_metadata(Tok(), train, 256)
    assert [x["blocks"] for x in lengths] == [1, 2]
    target = torch.arange(293)
    exact = compare_blocks(target.clone(), target, 256)
    assert all(b["exact"] for b in exact)
    output = target.clone()
    output[270] = -1
    compared = compare_blocks(output, target, 256)
    assert compared[0]["exact"] and not compared[1]["exact"]
    early = compare_blocks(target[:20], target, 256)
    assert not early[0]["exact"] and early[1]["available_tokens"] == 0


def test_second_block_gets_correct_prefix_and_full_canvas():
    model = tiny_model()
    observed = []
    sampled = []
    target = torch.tensor(list(range(4, 24)) + [1])

    class Tok:
        bos_token = "B"

        def encode(self, text, **kwargs):
            return [2, 3] if "user" in text else target.tolist()

        def decode(self, ids, **kwargs):
            return "decoded"

    args = SimpleNamespace(suite="length", sample_steps=2, seeds=[42])

    def recover(model, prompt, canvas, **kwargs):
        observed.append((prompt.clone(), canvas.clone()))
        return torch.zeros(1, len(canvas), 64)

    def one_block(model, prompt, **kwargs):
        sampled.append(prompt.clone())
        return torch.zeros(1, kwargs["canvas_size"], dtype=torch.long), torch.tensor(0.0)

    row = {"kind": "cross_block_text", "prompt": "P", "response": "R"}
    with (
        patch("run_single_sample_controls.denoise_logits", side_effect=recover),
        patch("run_single_sample_controls.sample_response", return_value=target.clone()) as rollout,
        patch("run_single_sample_controls.sample_canvas", side_effect=one_block),
    ):
        report = evaluate(model, Tok(), [row], [], args, 16)
    assert rollout.call_args.kwargs["max_canvases"] == 3
    assert len(observed) == 6 and all(len(c) == 16 for _, c in observed)
    torch.testing.assert_close(observed[3][0], torch.cat([torch.tensor([2, 3]), target[:16]]))
    assert report["recovery"][3]["tail_count"] == 11
    assert report["recovery"][3]["body_count"] == 5
    for item in report["recovery"]:
        assert item["tail_changed"] + item["tail_kept"] == item["tail_count"]
        assert 0 <= item["tail_restored"] <= item["tail_changed"]
        assert 0 <= item["tail_preserved"] <= item["tail_kept"]
    assert len(sampled) == 2
    torch.testing.assert_close(sampled[1], observed[3][0])
    assert [r["block"] for r in report["teacher_blocks"]] == [0, 1]
    assert all(r["exact"] for r in report["samples"][0]["blocks"])


def test_length_cli_failure_does_not_advance(tmp_path):
    out = tmp_path / "length"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_single_sample_controls.py"),
        "--tiny",
        "--suite",
        "length",
        "--out",
        str(out),
        "--steps-per-stage",
        "2",
        "--eval-every",
        "1",
        "--sample-steps",
        "2",
        "--seeds",
        "42",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failed_stage"] == 1
    assert not (out / "stage-2").exists()
    assert (out / "checkpoint-2/COMPLETE").exists()
    r = subprocess.run(cmd + ["--summary-only"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "FREE_BLOCKS" in r.stdout
