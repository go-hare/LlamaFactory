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
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file
from test_diffusion_ple import tiny_model


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_single_sample_controls import (
    clean_answer,
    evaluate,
    fresh_destination,
    load_model_weights,
    preflight,
    save_checkpoint,
    stable_pass,
    task_rows,
)
from validate_diffusion_conversion import ByteTokenizer, preserve_rng


def test_training_cases_have_distinct_answers_and_unseen_prompts():
    train, heldout = task_rows()
    assert [x["response"] for x in train] == ["5", "7", "9", "11"]
    assert len({x["prompt"] for x in train + heldout}) == 6
    assert len({x["response"] for x in train + heldout}) == 6
    assert clean_answer("55<turn|>") != "5"
    assert clean_answer("5<turn|>") == "5"


def test_stage_gate_requires_all_seeds_and_two_successive_evaluations():
    report = {"samples": [{"split": "train", "passed": True} for _ in range(4)]}
    report["samples"].append({"split": "heldout", "passed": False})
    assert stable_pass(report, 0) == 1
    assert stable_pass(report, 1) == 2
    report["samples"][2]["passed"] = False
    assert stable_pass(report, 1) == 0
    assert stable_pass({"samples": []}, 1) == 0


def test_gpu_rejected_before_model_load():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="No CUDA"):
            preflight()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(22 * 1024**3, 24 * 1024**3)),
        patch("torch.cuda.get_device_name", return_value="L4"),
    ):
        with pytest.raises(RuntimeError, match="No training started"):
            preflight()


@pytest.mark.parametrize("cpu_gib,local_gib,output_gib", [(1, 200, 200), (100, 1, 200), (100, 200, 1)])
def test_host_resources_rejected(cpu_gib, local_gib, output_gib, tmp_path):
    from types import SimpleNamespace

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(70 * 1024**3, 80 * 1024**3)),
        patch("torch.cuda.get_device_name", return_value="A100"),
        patch("psutil.virtual_memory", return_value=SimpleNamespace(available=cpu_gib * 1024**3)),
        patch(
            "shutil.disk_usage",
            side_effect=[SimpleNamespace(free=local_gib * 1024**3), SimpleNamespace(free=output_gib * 1024**3)],
        ),
    ):
        with pytest.raises(RuntimeError, match="No model download or training started"):
            preflight(out=tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_checkpoint_shards_and_strict_key_validation(tmp_path):
    from safetensors.torch import save_file

    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    original = model.save_pretrained
    with patch.object(model, "save_pretrained", side_effect=lambda path: original(path, max_shard_size="1KB")):
        save_checkpoint(model, ByteTokenizer(), optimizer, tmp_path, dict(global_step=1))
    path = tmp_path / "checkpoint-1"
    assert (path / "model.safetensors.index.json").exists()
    fresh = tiny_model()
    load_model_weights(fresh, path)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], value, rtol=0, atol=0)
    broken = tmp_path / "broken"
    broken.mkdir()
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del weights["model.language_model.per_layer_model_projection.weight"]
    save_file(weights, str(broken / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Invalid diagnostic weights"):
        load_model_weights(tiny_model(), broken)
    weights["model.language_model.per_layer_model_projection.weight"] = (
        model.model.language_model.per_layer_model_projection.weight.detach().clone()
    )
    weights["unexpected.weight"] = torch.zeros(1)
    save_file(weights, str(broken / "model.safetensors"))
    with pytest.raises(RuntimeError, match="unexpected"):
        load_model_weights(tiny_model(), broken)


def test_checkpoint_conflict_fails_without_writing(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    (tmp_path / "checkpoint-1").mkdir()
    with pytest.raises(FileExistsError, match="Nothing overwritten"):
        save_checkpoint(model, ByteTokenizer(), optimizer, tmp_path, dict(global_step=1))
    assert not (tmp_path / ".checkpoint-1.writing").exists()
    with pytest.raises(RuntimeError, match="nothing overwritten"):
        fresh_destination(tmp_path)


def test_reports_use_full_canvas_and_preserve_randomness():
    from types import SimpleNamespace

    model = tiny_model()

    # The real tokenizer fits short answers; this synthetic tokenizer emits ids
    # inside the tiny vocabulary and renders controlled sampler outputs.
    class Tokenizer:
        bos_token = "B"

        def encode(self, text, **kwargs):
            return [2, 3] if "user" in text else [4, 106 % 64]

        def decode(self, ids, **kwargs):
            return "5<turn|>"

    args = SimpleNamespace(sample_steps=2, seeds=[42, 43])
    observed = []
    import run_single_sample_controls as runner

    original = runner.denoise_logits

    def record(model, prompt, canvas, **kwargs):
        observed.append(len(canvas))
        return original(model, prompt, canvas, **kwargs)

    train, heldout = task_rows()
    state = torch.get_rng_state().clone()
    with (
        patch.object(runner, "denoise_logits", side_effect=record),
        patch.object(runner, "sample_response", return_value=torch.tensor([4, 1])),
    ):
        with preserve_rng():
            report = evaluate(model, Tokenizer(), train[:1], heldout[:1], args, 8)
    assert torch.equal(torch.get_rng_state(), state)
    assert observed == [8] * 6
    assert len(report["samples"]) == 4
    assert len(report["recovery"]) == 6
    assert report["samples"][0]["passed"]
    assert not model.training


def test_complete_checkpoint_can_restore_next_optimizer_update(tmp_path):
    from llamafactory.train.diffusion.loss import _get_conditioner

    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1.125e-4)
    tracked = model.model.language_model.layers[0].mlp.up_proj.weight
    tracked.grad = torch.ones_like(tracked)
    opt.step()
    state = dict(global_step=1, stage_index=0, stage_step=1, streak=0, completed=[], settings={})
    save_checkpoint(model, ByteTokenizer(), opt, tmp_path, state)
    path = tmp_path / "checkpoint-1"
    assert (path / "COMPLETE").exists()
    saved = torch.load(path / "diagnostic_state.pt", weights_only=False)
    fresh = tiny_model()
    load_model_weights(fresh, path)
    _get_conditioner(fresh).load_state_dict(load_file(str(path / "self_conditioner.safetensors")), strict=True)
    restored = torch.optim.AdamW(fresh.parameters(), lr=1.125e-4)
    restored.load_state_dict(saved["optimizer"])
    other = fresh.model.language_model.layers[0].mlp.up_proj.weight
    tracked.grad = torch.full_like(tracked, 0.5)
    other.grad = torch.full_like(other, 0.5)
    opt.step()
    restored.step()
    torch.testing.assert_close(tracked, other, rtol=0, atol=0)


def test_tiny_cli_stops_on_failed_stage_without_advancing(tmp_path):
    out = tmp_path / "run"
    command = [
        sys.executable,
        str(SCRIPTS / "run_single_sample_controls.py"),
        "--tiny",
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
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "stopped_at_failed_stage"
    assert summary["failed_stage"] == 1
    assert not (out / "stage-2").exists()
    assert (out / "checkpoint-2/COMPLETE").exists()
    second = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0
    assert "nothing overwritten" in second.stderr
    summary_run = subprocess.run(command + ["--summary-only"], capture_output=True, text=True, timeout=30)
    assert summary_run.returncode == 0
    assert "stopped_at_failed_stage" in summary_run.stdout
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    conflict = subprocess.run(
        command + ["--resume", str(out / "checkpoint-1")], capture_output=True, text=True, timeout=30
    )
    assert conflict.returncode != 0
    assert "fresh empty --out" in conflict.stderr
    assert before == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert not list(out.glob(".checkpoint-*.writing"))
    fork = tmp_path / "resumed"
    resume_command = command.copy()
    resume_command[resume_command.index("--out") + 1] = str(fork)
    resume_command += ["--resume", str(out / "checkpoint-1")]
    resumed = subprocess.run(resume_command, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    uninterrupted = load_file(str(out / "checkpoint-2/model.safetensors"))
    continued = load_file(str(fork / "checkpoint-2/model.safetensors"))
    assert uninterrupted.keys() == continued.keys()
    for key in uninterrupted:
        torch.testing.assert_close(uninterrupted[key], continued[key], rtol=0, atol=0)
