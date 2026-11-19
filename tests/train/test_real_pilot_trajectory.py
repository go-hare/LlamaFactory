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
import audit_real_pilot_trajectory as audit
from run_real_conversion_pilot import tiny_setup

from llamafactory.train.diffusion.rl import sample_canvas


def test_fixed_plan_includes_train_valid_first_canvases_and_longest_last():
    _, _, splits = tiny_setup()
    plan = audit.audit_plan(splits, 16)
    assert len(plan) == 10
    for split in ("train", "valid"):
        first = [p for p in plan if p["split"] == split and p["context"] == "prompt_only"]
        assert {p["bucket"] for p in first} == {0, 1, 2, 3}
        assert all(p["block"] == 0 for p in first)
        last = next(p for p in plan if p["split"] == split and p["context"] == "gold_previous_canvases")
        longest = max(range(4), key=lambda i: len(splits[split][i]["response_ids"]))
        assert last["row"] == longest and last["block"] == 3


@pytest.mark.parametrize("steps,seed", [(3, 100), (48, 101)])
def test_carried_trajectory_matches_production_output_and_preserves_weights(steps, seed):
    model = tiny_model()
    model.eval().requires_grad_(False)
    prompt, target = torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7, 1])
    before = {name: p.detach().clone() for name, p in model.state_dict().items()}
    torch.manual_seed(seed)
    with torch.inference_mode():
        expected, _ = sample_canvas(model, prompt, steps=steps, canvas_size=4, vocab_size=64)
        report = audit.trajectory(
            model, prompt, target, 4, steps=steps, seed=seed, self_conditioning=True, seconds=120
        )
        off = audit.trajectory(model, prompt, target, 4, steps=steps, seed=seed, self_conditioning=False, seconds=120)
    assert expected[0].tolist() == report["output_ids"]
    assert report["initial_ids"] == off["initial_ids"]
    assert report["used_steps"] == len(report["trace"]) == steps
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_zero_control_never_gets_signal_but_carried_uses_previous_prediction():
    model = tiny_model()
    prompt, target = torch.tensor([2, 3]), torch.tensor([5, 6, 7, 1])
    original = audit.denoise_step_logits
    signals = []

    def observed(model, history, canvas, **kwargs):
        signals.append(kwargs["sc_embeddings"].clone())
        return original(model, history, canvas, **kwargs)

    with patch.object(audit, "denoise_step_logits", side_effect=observed):
        audit.trajectory(model, prompt, target, 4, steps=3, seed=100, self_conditioning=True, seconds=120)
    assert not torch.count_nonzero(signals[0])
    assert all(torch.count_nonzero(signal) for signal in signals[1:])
    signals.clear()
    with patch.object(audit, "denoise_step_logits", side_effect=observed):
        audit.trajectory(model, prompt, target, 4, steps=3, seed=100, self_conditioning=False, seconds=120)
    assert all(not torch.count_nonzero(signal) for signal in signals)


def test_score_body_tail_and_early_stop_keep_sampled_state():
    model = tiny_model()
    prompt, target = torch.tensor([2, 3]), torch.tensor([5, 6, 1, 1])
    logits = torch.zeros(1, 4, 64)
    logits[0, torch.arange(4), target] = 10
    stats = audit.score(logits, torch.tensor([1, 5, 1, 1]), target, 2)
    assert stats["body"]["tokens"] == stats["tail"]["tokens"] == 2
    assert stats["body"]["argmax_reference_accuracy"] == 1
    assert stats["body"]["sample_reference_accuracy"] == 0
    assert stats["stop_before_reference"]
    torch.manual_seed(100)
    initial = torch.randint(0, 64, (1, 4))
    logits = torch.full((1, 4, 64), -100.0)
    logits.scatter_(2, initial[..., None], 100.0)
    sampled = (initial + 1) % 64
    with (
        patch.object(audit, "denoise_step_logits", return_value=logits),
        patch.object(audit, "_commit", return_value=(sampled, logits, torch.zeros(1, 4))),
    ):
        report = audit.trajectory(model, prompt, target, 2, steps=3, seed=100, self_conditioning=True, seconds=120)
    assert report["used_steps"] == 1 and report["stop_reason"] == "entropy_and_stability"
    assert report["output_ids"] == sampled[0].tolist()


def test_soft_budget_stops_between_steps_without_claiming_convergence():
    model = tiny_model()
    with patch.object(audit.time, "perf_counter", side_effect=[0.0, 2.0, 2.0]):
        result = audit.trajectory(
            model,
            torch.tensor([2, 3]),
            torch.tensor([5, 6, 7, 1]),
            4,
            steps=48,
            seed=100,
            self_conditioning=True,
            seconds=1,
        )
    assert result["used_steps"] == 1 and result["stop_reason"] == "time_budget"


def test_gold_target_never_leaks_into_first_canvas_inputs(tmp_path):
    model, tokenizer, splits = tiny_setup()
    plan = audit.audit_plan(splits, 16)
    observed = []

    def denoise(model, history, canvas, **kwargs):
        observed.append(history.tolist())
        return torch.zeros(1, 16, 259)

    with (
        patch.object(audit, "denoise_step_logits", side_effect=denoise),
        patch.object(audit, "denoise_logits", side_effect=denoise),
    ):
        records = audit.execute(model, tokenizer, splits, plan, tmp_path, 1, 100, 120, 16)
    assert len(records) == 20 and len(observed) == 30
    for item, offset in zip(plan, range(0, len(observed), 3)):
        row = splits[item["split"]][item["row"]]
        expected = row["prompt_ids"] + row["response_ids"][: item["block"] * 16]
        assert all(history == expected for history in observed[offset : offset + 3])
    summary = audit.summarize(records)
    assert summary["training_started"] is False and summary["trajectories"] == 20
    records[1]["initial_ids"][0] += 1
    with pytest.raises(ValueError, match="initial noise"):
        audit.summarize(records)


def test_cli_is_frozen_and_does_not_read_optimizer_state(tmp_path):
    training, out = tmp_path / "training", tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_real_conversion_pilot.py"),
            "--tiny",
            "--out",
            str(training),
            "--max-steps",
            "2",
            "--grad-accum",
            "1",
            "--eval-every",
            "1",
            "--generation-every",
            "1",
            "--sample-steps",
            "1",
            "--generation-canvases",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = training / "checkpoint-2"
    original = {
        str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    checkpoint.joinpath("diagnostic_state.pt").write_bytes(b"Must not deserialize this optimizer state")
    original["diagnostic_state.pt"] = hashlib.sha256(
        checkpoint.joinpath("diagnostic_state.pt").read_bytes()
    ).hexdigest()
    command = [
        sys.executable,
        str(SCRIPTS / "audit_real_pilot_trajectory.py"),
        "--tiny",
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["checkpoint_inventory_unchanged"] and summary["training_started"] is False
    assert summary["trajectories"] == 20
    settings = json.loads((out / "settings.json").read_text())
    assert settings["optimizer_created_or_loaded"] is False
    assert original == {
        str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0 and "nothing overwritten" in blocked.stderr
    with (
        patch.object(sys, "argv", ["audit", "--out", str(out), "--summary-only"]),
        patch("torch.cuda.is_available", side_effect=AssertionError("GPU checked")),
    ):
        audit.main()
    guarded_out = tmp_path / "guarded-audit"
    with (
        patch.object(sys, "argv", ["audit", "--tiny", "--checkpoint", str(checkpoint), "--out", str(guarded_out)]),
        patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer created")),
        patch("torch.load", side_effect=AssertionError("Optimizer checkpoint deserialized")),
        patch.object(torch.Tensor, "backward", side_effect=AssertionError("Backward called")),
    ):
        audit.main()
    assert json.loads((guarded_out / "summary.json").read_text())["trajectories"] == 20
    training.joinpath("settings.json").write_text(
        json.dumps(dict(json.loads((training / "settings.json").read_text()), code_sha256="wrong"))
    )
    args = ["audit", "--tiny", "--checkpoint", str(checkpoint), "--out", str(tmp_path / "rejected")]
    with patch.object(sys, "argv", args), patch.object(audit, "Gemma4ForConditionalGeneration") as loader:
        with pytest.raises(RuntimeError, match="source mismatch"):
            audit.main()
    loader.from_pretrained.assert_not_called()
    assert not (tmp_path / "rejected").exists()
