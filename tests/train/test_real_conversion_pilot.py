# ruff: noqa: E402

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_real_conversion_pilot as pilot
from validate_diffusion_conversion import preserve_rng


def test_epoch_schedule_uses_each_row_once_without_global_rng_changes():
    before = torch.get_rng_state().clone()
    first = [pilot.schedule_index(i, 512) for i in range(512)]
    second = [pilot.schedule_index(512 + i, 512) for i in range(512)]
    assert sorted(first) == sorted(second) == list(range(512))
    assert first != second
    assert first[107:] == [pilot.schedule_index(i, 512) for i in range(107, 512)]
    assert torch.equal(before, torch.get_rng_state())


def test_recovery_uses_fixed_full_canvas_and_generation_has_no_gold_prefix(tmp_path):
    model, tokenizer, splits = pilot.tiny_setup()
    plan = pilot.fixed_plan(splits, 16)
    settings = {"canvas": 16, "sample_steps": 2, "generation_canvases": 1}
    calls, samples = [], []

    def recover(model, history, noisy, **kwargs):
        calls.append((history.clone(), noisy.clone()))
        return torch.zeros(1, 16, 259)

    def generate(model, prompt, **kwargs):
        samples.append(prompt.clone())
        return torch.tensor([123, 1])

    model.train()
    before = torch.get_rng_state().clone()
    with (
        patch.object(pilot, "denoise_logits", side_effect=recover),
        patch.object(pilot, "sample_response", side_effect=generate),
        preserve_rng(),
    ):
        report = pilot.evaluate(model, tokenizer, splits, plan, settings, tmp_path, 0, True)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert len(calls) == len(plan["recovery"]) * 3
    assert all(len(noisy) == 16 for _, noisy in calls)
    for index, item in enumerate(plan["recovery"]):
        row = splits[item["split"]][item["row"]]
        expected = row["prompt_ids"] + row["response_ids"][: item["block"] * 16]
        assert calls[index * 3][0].tolist() == expected
    for prompt, item in zip(samples, plan["generation"]):
        assert prompt.tolist() == splits[item["split"]][item["row"]]["prompt_ids"]
    assert len(report["samples"]) == 4
    assert len(report["aggregate"]) == 6
    tail = [r for r in report["recovery"] if r["tail_count"]]
    assert tail
    for item in report["recovery"]:
        assert item["body_count"] + item["tail_count"] == 16
        for region in ("body", "tail"):
            assert item[f"{region}_changed"] + item[f"{region}_kept"] == item[f"{region}_count"]
    assert (tmp_path / "eval-0000.json").exists()
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 4


def test_cap_and_reference_exact_are_reported_separately():
    _, tokenizer, splits = pilot.tiny_setup()
    row = dict(splits["valid"][0], response="x<turn|>", response_ids=[123, 1])
    stopped = pilot.score_sample(row, torch.tensor([123, 1]), tokenizer, 16, 1)
    assert stopped["reference_exact"] and not stopped["hit_canvas_cap"]
    capped = pilot.score_sample(row, torch.full((16,), 123), tokenizer, 16, 1)
    assert capped["hit_canvas_cap"] and capped["stop_reason"] == "canvas_budget"
    assert not capped["reference_exact"]
    long_row = dict(row, response_ids=list(range(17)))
    assert pilot.score_sample(long_row, torch.tensor([123, 1]), tokenizer, 16, 1)["reference_exceeds_generation_cap"]


def test_generation_uses_own_previous_canvas_and_marks_soft_time_limit():
    model, tokenizer, splits = pilot.tiny_setup()
    prompt = torch.tensor([7, 8])
    settings = {"canvas": 16, "sample_steps": 1, "generation_canvases": 18, "generation_seconds": 120}
    first = torch.full((16,), 123)
    with patch.object(pilot, "sample_response", side_effect=[first, torch.tensor([9, 1])]) as sampler:
        output, reason = pilot.sample_pilot_response(model, prompt, settings)
    assert reason == "stop_token" and len(output) == 18
    assert sampler.call_args_list[1].args[1].tolist() == prompt.tolist() + first.tolist()
    settings["generation_seconds"] = 1
    with (
        patch.object(pilot, "sample_response", return_value=first) as sampler,
        patch("run_real_conversion_pilot.time.perf_counter", side_effect=[0.0, 0.5, 1.5]),
    ):
        output, reason = pilot.sample_pilot_response(model, prompt, settings)
    assert sampler.call_count == 1 and reason == "time_budget"
    record = pilot.score_sample(splits["valid"][0], output, tokenizer, 16, 18, reason)
    assert record["hit_time_budget"] and not record["hit_canvas_cap"] and not record["has_stop"]


def test_pretokenized_ids_must_match_tokenizer_before_model_load():
    class Tok:
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self):
            return 259

        def encode(self, text, **kwargs):
            return [106] if text == "<turn|>" else [8]

    row = {"prompt": "P", "response": "R", "prompt_ids": [8], "response_ids": [8]}
    pilot.verify_tokens({"train": [row]}, Tok())
    with pytest.raises(ValueError, match="differ"):
        pilot.verify_tokens({"train": [dict(row, response_ids=[9])]}, Tok())
    with pytest.raises(ValueError, match="outside"):
        pilot.verify_tokens({"train": [dict(row, response_ids=[True])]}, Tok())


def test_data_integrity_and_cross_split_duplicates(tmp_path):
    import hashlib

    from prepare_real_conversion_pilot import write_dataset
    from test_real_pilot_data import CharacterTokenizer, sources

    train, valid = sources(tmp_path)
    out = tmp_path / "data"
    write_dataset(train, valid, CharacterTokenizer(), out, 1, 1)
    splits, manifest = pilot.load_data(out)
    assert len(splits["train"]) == len(splits["valid"]) == 4
    path = out / "train.jsonl"
    content = path.read_text()
    path.write_text(content + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.load_data(out)
    rows = content.splitlines()
    rows[0] = json.dumps(splits["valid"][0])
    path.write_text("\n".join(rows) + "\n")
    manifest["splits"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.load_data(out)


def test_tiny_budget_checkpoint_resume_and_rejections(tmp_path):
    out = tmp_path / "pilot"
    command = [
        sys.executable,
        str(SCRIPTS / "run_real_conversion_pilot.py"),
        "--tiny",
        "--out",
        str(out),
        "--max-steps",
        "2",
        "--eval-every",
        "1",
        "--generation-every",
        "2",
        "--grad-accum",
        "2",
        "--sample-steps",
        "1",
        "--generation-canvases",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "budget_complete_no_full_training"
    assert summary["global_step"] == 2
    assert sum(summary["sampled_block_counts"].values()) == 4
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"
    state = torch.load(out / "checkpoint-2/diagnostic_state.pt", map_location="cpu", weights_only=False)
    weights = load_file(str(out / "checkpoint-2/model.safetensors"))
    assert "model.language_model.layers.0.layer_scalar" in weights
    assert sum(item.get("exp_avg", torch.empty(0)).numel() == 1 for item in state["optimizer"]["state"].values()) == 2
    assert len((out / "train.jsonl").read_text().splitlines()) == 2
    assert len((out / "samples.jsonl").read_text().splitlines()) == 8
    continued = tmp_path / "continued"
    cmd = command.copy()
    cmd[cmd.index("--out") + 1] = str(continued)
    cmd += ["--resume", str(out / "checkpoint-1")]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = load_file(str(out / "checkpoint-2/model.safetensors"))
    actual = load_file(str(continued / "checkpoint-2/model.safetensors"))
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    old = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "nothing overwritten" in result.stderr
    assert old == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    args = ["pilot", "--out", str(out), "--summary-only"]
    with (
        patch.object(sys, "argv", args),
        patch("torch.cuda.is_available", side_effect=AssertionError("Summary must not inspect GPU")),
    ):
        pilot.main()
    mismatch = command.copy()
    mismatch[mismatch.index("--out") + 1] = str(tmp_path / "mismatch")
    mismatch[mismatch.index("--grad-accum") + 1] = "1"
    result = subprocess.run(
        mismatch + ["--resume", str(out / "checkpoint-1")], capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0 and "settings/code/data differ" in result.stderr
    assert not (tmp_path / "mismatch").exists()
