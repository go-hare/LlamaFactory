"""Tests for Qwen3 AR-to-diffusion conversion compatibility and pilot script."""
# ruff: noqa: E402

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_qwen3_conversion_pilot as qpilot

from llamafactory.train.diffusion.adapter import Gemma4Adapter, StandardCausalAdapter, get_diffusion_adapter
from llamafactory.train.diffusion.loss import (
    _embed_distribution,
    _get_conditioner,
    _token_embedding,
    denoise_step_logits,
    diffusion_batch_step,
    prepare_conversion_training,
)
from llamafactory.train.diffusion.objective import (
    MaskTokenCorruption,
    resolve_corruption,
    resolve_objective,
)
from llamafactory.train.diffusion.rl import _ddim_step


def tiny_qwen3(attn: str = "eager") -> Qwen3ForCausalLM:
    torch.manual_seed(11)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    model.set_attn_implementation(attn)
    prepare_conversion_training(model)
    return model


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_qwen3_diffusion_bidirectionality_and_embed_scale(attn):
    model = tiny_qwen3(attn)
    adapter = get_diffusion_adapter(model)
    assert isinstance(adapter, StandardCausalAdapter)
    assert adapter.embed_scale(model) == 1.0
    logits_in = torch.zeros(1, 3, 64)
    emb = _embed_distribution(_token_embedding(model), logits_in, scale=adapter.embed_scale(model))
    expected_emb = torch.softmax(logits_in, dim=-1) @ _token_embedding(model).weight
    torch.testing.assert_close(emb, expected_emb)

    prompt = torch.tensor([2, 3, 4])
    canvas = torch.tensor([5, 6, 7, 8])
    changed = torch.tensor([5, 6, 7, 29])
    zero = torch.zeros(1, 4, 16)
    with torch.no_grad():
        a = denoise_step_logits(model, prompt, canvas, sc_embeddings=zero, pad_id=0)
        b = denoise_step_logits(model, torch.tensor([2, 20, 21]), canvas, sc_embeddings=zero, pad_id=0)
        c = denoise_step_logits(model, prompt, changed, sc_embeddings=zero, pad_id=0)
    assert float((a - b).abs().max()) > 1e-5
    assert float((a[:, 0] - c[:, 0]).abs().max()) > 1e-5


def test_qwen3_diffusion_backward_updates_self_conditioner():
    model = tiny_qwen3().train()
    loss, _ = diffusion_batch_step(
        model,
        [torch.tensor([2, 3, 4])],
        [torch.tensor([10, 11, 12, 13, 14])],
        canvas_size=4,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    cond = _get_conditioner(model)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in cond.parameters())


def test_pluggable_mask_corruption_and_corrupted_only_objective():
    model = tiny_qwen3().train()
    mask_id = 63
    loss, _ = diffusion_batch_step(
        model,
        [torch.tensor([2, 3, 4])],
        [torch.tensor([10, 11, 12, 13, 14])],
        canvas_size=4,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
        loss_mode="corrupted-only",
        corruption_mode="mask",
        mask_id=mask_id,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert isinstance(resolve_corruption("mask", mask_id=mask_id), MaskTokenCorruption)
    assert resolve_objective("corrupted-only").name == "corrupted-only"


def test_ddim_high_noise_stop_protection():
    vocab_size = 64
    canvas = torch.full((1, 512), 1, dtype=torch.long)
    logits = torch.full((1, 512, vocab_size), -1e4)
    logits[:, :, 1] = 1e4  # confident stop token at high noise
    torch.manual_seed(0)
    unprotected, _, _ = _ddim_step(
        logits, canvas, time_now=0.9, time_next=0.85, temperature=0.7, vocab_size=vocab_size
    )
    torch.manual_seed(0)
    protected, _, _ = _ddim_step(
        logits,
        canvas,
        time_now=0.9,
        time_next=0.85,
        temperature=0.7,
        vocab_size=vocab_size,
        protect_stop_tokens=(1,),
        protect_stop_time=0.75,
    )
    assert (protected == 1).float().mean() < (unprotected == 1).float().mean() - 0.15


def test_gemma_to_qwen3_text_conversion():
    p = "<bos><|turn>system\n<|think|>\nsys<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
    r = "<|channel>thought\nplan<channel|>answer<turn|>"
    qp, qr = qpilot.gemma_to_qwen3_texts(p, r)
    assert "<|im_start|>system\nsys<|im_end|>\n" in qp
    assert qp.endswith("<|im_start|>assistant\n")
    assert qr == "<think>\nplan\n</think>\nanswer<|im_end|>"


def test_qwen3_pilot_tiny_full_and_resume(tmp_path):
    out = tmp_path / "qwen_full"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--lora-rank",
        "0",
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
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"
    assert (out / "eval-0002.json").is_file()

    resumed = tmp_path / "qwen_resumed"
    res2 = subprocess.run(
        cmd[: cmd.index("--out") + 1] + [str(resumed)] + cmd[cmd.index("--out") + 2 :] + ["--resume", str(out / "checkpoint-1")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (resumed / "checkpoint-2/COMPLETE").read_text() == "2"


def test_compute_lr_cosine_warmup_and_resume():
    # Fresh run: linear warmup -> cosine decay to min_lr_ratio
    assert abs(qpilot.compute_lr(25, max_steps=1000, peak_lr=2e-4, warmup_steps=50, min_lr_ratio=0.1) - 1e-4) < 1e-9
    assert abs(qpilot.compute_lr(50, max_steps=1000, peak_lr=2e-4, warmup_steps=50, min_lr_ratio=0.1) - 2e-4) < 1e-9
    mid = qpilot.compute_lr(525, max_steps=1000, peak_lr=2e-4, warmup_steps=50, min_lr_ratio=0.1)
    assert abs(mid - 1.1e-4) < 1e-9
    end = qpilot.compute_lr(1000, max_steps=1000, peak_lr=2e-4, warmup_steps=50, min_lr_ratio=0.1)
    assert abs(end - 2e-5) < 1e-9

    # Resumed run: decays smoothly from peak_lr to min_lr_ratio over [start_step, max_steps]
    r_start = qpilot.compute_lr(5121, max_steps=7168, peak_lr=6e-5, min_lr_ratio=0.1, start_step=5120)
    r_mid = qpilot.compute_lr(6144, max_steps=7168, peak_lr=6e-5, min_lr_ratio=0.1, start_step=5120)
    r_end = qpilot.compute_lr(7168, max_steps=7168, peak_lr=6e-5, min_lr_ratio=0.1, start_step=5120)
    assert r_start <= 6e-5 and r_start > 5.9e-5
    assert abs(r_mid - 3.3e-5) < 1e-9
    assert abs(r_end - 6e-6) < 1e-9


def test_qwen3_micro_batch_and_logits_to_keep_end_to_end(tmp_path):
    out = tmp_path / "qwen_batched"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--lora-rank",
        "0",
        "--out",
        str(out),
        "--max-steps",
        "2",
        "--eval-every",
        "2",
        "--generation-every",
        "2",
        "--micro-batch",
        "2",
        "--grad-accum",
        "2",
        "--loss-mode",
        "body-tail-balanced",
        "--lr-scheduler",
        "cosine",
        "--sample-steps",
        "2",
        "--generation-canvases",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"
    lines = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert len(lines[0]["rows"]) == 4
