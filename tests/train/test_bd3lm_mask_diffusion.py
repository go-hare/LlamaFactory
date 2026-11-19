"""Tests for BD3-LMs style [MASK] absorbing diffusion, CorruptedOnly loss, and clipped time schedules."""
# ruff: noqa: E402

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_qwen3_diffusion import tiny_qwen3

from llamafactory.train.diffusion.loss import diffusion_batch_step
from llamafactory.train.diffusion.objective import MaskTokenCorruption, UniformNoiseCorruption
from llamafactory.train.diffusion.rl import _mask_unmask_step, sample_canvas, sample_response


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_clipped_time_schedule_stays_within_bounds():
    clean = torch.arange(32, dtype=torch.long)
    mask_proc = MaskTokenCorruption(mask_id=63)
    uni_proc = UniformNoiseCorruption()
    for _ in range(25):
        ms = mask_proc.corrupt(clean, vocab_size=64, time_min=0.3, time_max=0.75)
        us = uni_proc.corrupt(clean, vocab_size=64, time_min=0.3, time_max=0.75)
        assert 0.3 <= ms.time <= 0.75
        assert 0.3 <= us.time <= 0.75
        assert torch.all(ms.noisy[ms.corrupted] == 63)
        assert torch.equal(ms.noisy[~ms.corrupted], clean[~ms.corrupted])

    with pytest.raises(ValueError, match="Invalid clipped time interval"):
        mask_proc.corrupt(clean, vocab_size=64, time_min=0.8, time_max=0.4)


def test_mask_unmask_step_enforces_subs_and_locks_unmasked_tokens():
    torch.manual_seed(42)
    mask_id = 63
    vocab_size = 64
    canvas = torch.tensor([[10, mask_id, mask_id, mask_id, 20, mask_id]])
    # Even if logits strongly favor mask_id, SUBS zeroes out mask_id probability.
    logits = torch.full((1, 6, vocab_size), -10.0)
    logits[:, :, mask_id] = 100.0
    logits[:, 1, 5] = 20.0
    logits[:, 2, 6] = 10.0

    nxt, shaped, _ = _mask_unmask_step(
        logits,
        canvas,
        mask_id=mask_id,
        remaining_steps=2,
        temperature=0.7,
    )
    assert shaped[:, :, mask_id].isneginf().all()
    # Already-unmasked positions 0 and 4 must stay locked at 10 and 20.
    assert int(nxt[0, 0]) == 10 and int(nxt[0, 4]) == 20
    # 4 masked tokens over 2 remaining steps -> ceil(4/2) = 2 tokens unmasked, none equal to mask_id.
    assert int((nxt == mask_id).sum()) == 2
    assert int(nxt[0, 1]) == 5


def test_mask_sampler_completes_without_leaving_masks():
    model = tiny_qwen3().eval()
    mask_id = 63
    prompt = torch.tensor([2, 3, 4])
    torch.manual_seed(17)
    with torch.no_grad():
        canvas, ent = sample_canvas(
            model,
            prompt,
            steps=4,
            canvas_size=8,
            vocab_size=64,
            sampler="mask",
            mask_id=mask_id,
            early_stop=False,
        )
        out = sample_response(
            model,
            prompt,
            max_canvases=2,
            steps=4,
            canvas_size=8,
            vocab_size=64,
            sampler="mask",
            mask_id=mask_id,
            protect_stop_time=0.75,
        )
    assert not bool((canvas == mask_id).any())
    assert not bool((out == mask_id).any())
    assert torch.isfinite(ent)


def test_bd3lm_mask_clipped_batch_step_backward():
    model = tiny_qwen3().train()
    loss, parts = diffusion_batch_step(
        model,
        [torch.tensor([2, 3, 4])],
        [torch.tensor([10, 11, 12, 13, 14])],
        canvas_size=4,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
        time_min=0.25,
        time_max=0.85,
        diagnostics=True,
        loss_mode="corrupted-only-weighted",
        corruption_mode="mask",
        mask_id=63,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.25 <= parts["rows"][0]["time"] <= 0.85


def test_qwen3_pilot_tiny_bd3lm_mask_mode_end_to_end(tmp_path):
    out = tmp_path / "bd3lm_mask"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--lora-rank",
        "0",
        "--corruption-mode",
        "mask",
        "--loss-mode",
        "corrupted-only",
        "--sampler",
        "mask",
        "--time-min",
        "0.25",
        "--time-max",
        "0.85",
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
        "2",
        "--generation-canvases",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    settings = json.loads((out / "settings.json").read_text())
    assert settings["corruption_mode"] == "mask"
    assert settings["loss_mode"] == "corrupted-only"
    assert settings["sampler"] == "mask"
    assert settings["time_min"] == 0.25 and settings["time_max"] == 0.85
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"

    resumed = tmp_path / "bd3lm_mask_resumed"
    res2 = subprocess.run(
        cmd[: cmd.index("--out") + 1] + [str(resumed)] + cmd[cmd.index("--out") + 2 :] + ["--resume", str(out / "checkpoint-1")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (resumed / "checkpoint-2/COMPLETE").read_text() == "2"

    eval_out = tmp_path / "bd3lm_remdm_eval"
    eval_cmd = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--out",
        str(eval_out),
        "--lora-rank",
        "0",
        "--corruption-mode",
        "mask",
        "--loss-mode",
        "canvas-mean",
        "--sampler",
        "remdm",
        "--remask-eta",
        "0.1",
        "--sample-steps",
        "4",
        "--generation-canvases",
        "1",
        "--resume",
        str(out / "checkpoint-2"),
        "--eval-only",
    ]
    res3 = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=120)
    assert res3.returncode == 0, res3.stdout + res3.stderr
    assert "QWEN3_PILOT_EVAL_DONE" in res3.stdout
    assert (eval_out / "eval-0002.json").is_file()


def test_remdm_step_remasks_disagreed_tokens_and_finishes_clean():
    from llamafactory.train.diffusion.rl import _remdm_step

    torch.manual_seed(0)
    mask_id = 15
    canvas = torch.tensor([[3, 4, mask_id, mask_id]])
    logits = torch.full((1, 4, 16), -20.0)
    logits[0, :, 7] = 10.0  # Strongly prefers token 7 everywhere (disagrees with 3 and 4)
    logits[0, :, mask_id] = 30.0  # SUBS must zero this out

    next_canvas, shaped, _ = _remdm_step(
        logits,
        canvas,
        mask_id=mask_id,
        time_now=0.5,
        time_next=0.25,
        temperature=1.0,
        remask_eta=1.0,
        is_final_step=False,
    )
    assert shaped[:, :, mask_id].isneginf().all()
    # Disagreed unmasked tokens (3, 4) can be remasked back to mask_id.
    assert bool((next_canvas[0, :2] == mask_id).any())

    final_canvas, _, _ = _remdm_step(
        logits,
        next_canvas,
        mask_id=mask_id,
        time_now=0.25,
        time_next=0.0,
        temperature=1.0,
        remask_eta=0.5,
        is_final_step=True,
    )
    assert not bool((final_canvas == mask_id).any())
    assert bool((final_canvas == 7).sum() >= 2)


def test_rcd_residual_signal_gates_by_entropy_and_masked_positions():
    from llamafactory.train.diffusion.loss import rcd_residual_signal

    emb = torch.nn.Embedding(16, 8)
    mask_id = 15
    canvas = torch.tensor([[3, mask_id, mask_id]])
    logits = torch.zeros(1, 3, 16)
    logits[0, 0, 3] = 20.0  # Unmasked -> must be zeroed out regardless of logits
    logits[0, 1, 7] = 20.0  # Masked & low entropy -> alpha ~ 1.0 (strong residual)
    # Position 2 is uniform -> high entropy -> alpha ~ 0.0 (weak residual)

    sig = rcd_residual_signal(emb, logits, canvas, mask_id=mask_id, scale=1.0)
    assert torch.allclose(sig[0, 0], torch.zeros(8), atol=1e-6)
    assert sig[0, 1].norm() > sig[0, 2].norm() * 5.0


def test_qwen3_pilot_tiny_rcd_stage2_and_eval(tmp_path):
    stage1 = tmp_path / "stage1"
    cmd1 = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--lora-rank",
        "0",
        "--corruption-mode",
        "mask",
        "--loss-mode",
        "canvas-mean",
        "--sampler",
        "mask",
        "--out",
        str(stage1),
        "--max-steps",
        "1",
        "--eval-every",
        "1",
        "--generation-every",
        "1",
        "--grad-accum",
        "1",
        "--sample-steps",
        "2",
        "--generation-canvases",
        "1",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
    assert res1.returncode == 0, res1.stdout + res1.stderr

    stage2 = tmp_path / "stage2_rcd"
    cmd2 = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--lora-rank",
        "0",
        "--corruption-mode",
        "mask",
        "--loss-mode",
        "canvas-mean",
        "--residual-mode",
        "rcd",
        "--sampler",
        "rcd",
        "--remask-eta",
        "0.15",
        "--resume",
        str(stage1 / "checkpoint-1"),
        "--out",
        str(stage2),
        "--max-steps",
        "2",
        "--eval-every",
        "2",
        "--generation-every",
        "2",
        "--grad-accum",
        "1",
        "--sample-steps",
        "4",
        "--generation-canvases",
        "1",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (stage2 / "checkpoint-2/COMPLETE").read_text() == "2"
    settings = json.loads((stage2 / "settings.json").read_text())
    assert settings["residual_mode"] == "rcd"
    assert settings["sampler"] == "rcd"
