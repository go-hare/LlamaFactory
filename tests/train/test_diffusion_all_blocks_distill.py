"""Mathematical parity, zero-leakage, and AR-teacher KL distillation tests for all-blocks training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llamafactory.train.diffusion.loss import (
    _build_all_blocks_mask,
    _compute_ar_teacher_topk,
    _diffusion_forward,
    _gemma_masks,
    _topk_forward_kl,
    denoise_step_logits,
    diffusion_all_blocks_step,
    prepare_conversion_training,
)


def _tiny_qwen3(vocab_size: int = 128, seed: int = 7) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg).to(torch.float32)
    prepare_conversion_training(model)
    model.eval()
    return model


def test_all_blocks_matches_single_block_forwards_exactly():
    """Every noisy block in one staircase forward pass must match running that block individually."""
    model = _tiny_qwen3()
    C = 8
    pad_id = 0

    # Two ragged rows: row 0 has prompt len 5 and 3 blocks; row 1 has prompt len 9 and 2 blocks.
    p0 = torch.tensor([11, 12, 13, 14, 15], dtype=torch.long)
    clean0 = torch.tensor(list(range(20, 20 + 3 * C)), dtype=torch.long)
    noisy0 = torch.tensor(list(range(60, 60 + 3 * C)), dtype=torch.long)

    p1 = torch.tensor([31, 32, 33, 34, 35, 36, 37, 38, 39], dtype=torch.long)
    clean1 = torch.tensor(list(range(40, 40 + 2 * C)), dtype=torch.long)
    noisy1 = torch.tensor(list(range(90, 90 + 2 * C)), dtype=torch.long)

    # Clean prefixes drop the last block of each row: row 0 keeps 2 blocks, row 1 keeps 1 block.
    pref0 = torch.cat([p0, clean0[: 2 * C]])
    pref1 = torch.cat([p1, clean1[: 1 * C]])
    max_clean = max(pref0.numel(), pref1.numel())
    max_noisy = max(noisy0.numel(), noisy1.numel())
    total = max_clean + max_noisy

    tokens = torch.full((2, total), pad_id, dtype=torch.long)
    clean_valid = torch.zeros((2, max_clean), dtype=torch.bool)
    noisy_valid = torch.zeros((2, max_noisy), dtype=torch.bool)
    pos_ids = torch.zeros((2, total), dtype=torch.long)

    for b, (pref, p, nseq) in enumerate(((pref0, p0, noisy0), (pref1, p1, noisy1))):
        clen, nlen = pref.numel(), nseq.numel()
        tokens[b, :clen] = pref
        clean_valid[b, :clen] = True
        pos_ids[b, :clen] = torch.arange(clen)
        tokens[b, max_clean : max_clean + nlen] = nseq
        noisy_valid[b, :nlen] = True
        pos_ids[b, max_clean : max_clean + nlen] = torch.arange(nlen) + p.numel()

    allowed = _build_all_blocks_mask(
        clean_valid,
        noisy_valid,
        prompt_lengths=[p0.numel(), p1.numel()],
        start_blocks=[0, 0],
        num_sel_blocks=[3, 2],
        canvas_size=C,
    )
    masks = _gemma_masks(allowed, torch.float32, valid=torch.cat([clean_valid, noisy_valid], dim=1), model=model)
    zero_sc = torch.zeros((2, max_noisy, 32), dtype=torch.float32)

    with torch.no_grad():
        batched = _diffusion_forward(model, tokens, zero_sc, max_clean, masks, pos_ids, logits_to_keep=max_noisy).logits

        # Compare every block of both rows against standalone single-block denoise_step_logits.
        zero_one = torch.zeros((1, C, 32), dtype=torch.float32)
        for b, (p, clean, noisy, n_blk) in enumerate(((p0, clean0, noisy0, 3), (p1, clean1, noisy1, 2))):
            for k in range(n_blk):
                hist = torch.cat([p, clean[: k * C]]) if k > 0 else p
                ref = denoise_step_logits(model, hist, noisy[k * C : (k + 1) * C], sc_embeddings=zero_one, pad_id=pad_id)[0]
                got = batched[b, k * C : (k + 1) * C]
                max_diff = (got - ref).abs().max().item()
                assert max_diff < 1e-4, f"Row {b} block {k} diverged from single-block forward: max_diff={max_diff}"


def test_all_blocks_zero_future_and_current_clean_leakage():
    """Mutating clean block k (or later) must have EXACTLY 0.0 effect on noisy block k."""
    model = _tiny_qwen3()
    C = 8
    p = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    clean_a = torch.tensor(list(range(10, 10 + 2 * C)), dtype=torch.long)
    clean_b = clean_a.clone()
    # Mutate clean block 1 (indices C..2C-1): noisy block 0 and noisy block 1 MUST NOT see it!
    clean_b[C:] = 99
    noisy = torch.tensor(list(range(50, 50 + 3 * C)), dtype=torch.long)

    def _run(clean_pref):
        pref = torch.cat([p, clean_pref])
        max_clean, max_noisy = pref.numel(), noisy.numel()
        tokens = torch.cat([pref, noisy]).unsqueeze(0)
        c_val = torch.ones((1, max_clean), dtype=torch.bool)
        n_val = torch.ones((1, max_noisy), dtype=torch.bool)
        pos = torch.cat([torch.arange(max_clean), torch.arange(max_noisy) + p.numel()]).unsqueeze(0)
        allowed = _build_all_blocks_mask(c_val, n_val, [p.numel()], [0], [3], C)
        masks = _gemma_masks(allowed, torch.float32, valid=torch.cat([c_val, n_val], dim=1), model=model)
        zero_sc = torch.zeros((1, max_noisy, 32), dtype=torch.float32)
        with torch.no_grad():
            return _diffusion_forward(model, tokens, zero_sc, max_clean, masks, pos, logits_to_keep=max_noisy).logits[0]

    out_a = _run(clean_a)
    out_b = _run(clean_b)
    # Noisy block 0 (0:C) and noisy block 1 (C:2C) cannot see clean block 1 -> exact 0.0 diff!
    assert (out_a[: 2 * C] - out_b[: 2 * C]).abs().max().item() == 0.0
    # Noisy block 2 (2C:3C) DOES see clean block 1 -> must differ!
    assert (out_a[2 * C :] - out_b[2 * C :]).abs().max().item() > 1e-3


def test_topk_kl_properties_and_ar_teacher_body_only_masking():
    """Top-K forward KL is >= 0, zero at exact match, and ignores eos-padded tail tokens."""
    torch.manual_seed(0)
    t_logits = torch.randn(10, 64)
    top_vals, top_idx = torch.topk(t_logits, k=16, dim=-1)
    top_p = torch.softmax(top_vals, dim=-1)

    # Non-negativity on random student logits.
    s_logits = torch.randn(10, 64)
    kl_rand = _topk_forward_kl(s_logits, top_p, top_idx, kl_temp=1.0)
    assert (kl_rand >= -1e-6).all()
    assert kl_rand.mean().item() > 0.1

    # Exact match on top-K support with -inf elsewhere -> 0.0 KL.
    exact = torch.full((10, 64), -1e9)
    exact.scatter_(-1, top_idx, top_vals)
    kl_exact = _topk_forward_kl(exact, top_p, top_idx, kl_temp=1.0)
    assert kl_exact.abs().max().item() < 1e-5

    # Full step with frozen AR teacher: gradients flow to student only, teacher stays untouched.
    student = _tiny_qwen3(seed=11)
    teacher = _tiny_qwen3(seed=22)
    for param in teacher.parameters():
        param.requires_grad_(False)
    student.train()

    prompts = [torch.tensor([10, 11, 12, 13]), torch.tensor([14, 15, 16, 17, 18])]
    # Row 0 has 10 body tokens (6 tail EOS pads in block 0); row 1 spans 2 blocks (20 body tokens).
    responses = [torch.tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 1]), torch.tensor(list(range(30, 49)) + [1])]

    torch.manual_seed(123)
    loss, parts = diffusion_all_blocks_step(
        student,
        prompts,
        responses,
        canvas_size=16,
        num_canvases=4,
        pad_id=0,
        eos_id=1,
        diagnostics=True,
        max_train_blocks=4,
        ar_teacher=teacher,
        kl_weight=0.5,
        kl_topk=32,
        kl_temp=1.0,
    )
    assert torch.isfinite(loss)
    assert parts["trained_blocks"] == 3  # 1 block for row 0 + 2 blocks for row 1
    assert parts["trained_tokens"] == 48
    assert parts["kl_loss"] > 0.0
    assert parts["ce_loss"] > 0.0
    assert abs(parts["decoder_loss"] - (parts["ce_loss"] + 0.5 * parts["kl_loss"])) < 1e-4

    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_all_blocks_cli_tiny_smoke(tmp_path: Path):
    """End-to-end smoke test of run_qwen3_conversion_pilot.py with all-blocks + AR teacher KL."""
    out_dir = tmp_path / "all_blocks_run"
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "scripts/run_qwen3_conversion_pilot.py"),
        "--tiny",
        "--out",
        str(out_dir),
        "--training-mode",
        "all-blocks",
        "--max-train-blocks",
        "4",
        "--ar-teacher",
        "tiny",
        "--kl-weight",
        "0.5",
        "--kl-topk",
        "32",
        "--data-offset",
        "1",
        "--lora-rank",
        "4",
        "--max-steps",
        "2",
        "--micro-batch",
        "2",
        "--grad-accum",
        "1",
        "--eval-every",
        "2",
        "--generation-every",
        "2",
        "--sample-steps",
        "4",
        "--sampler",
        "entropy",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "AR_TEACHER_LOADED" in res.stdout
    assert "QWEN3_PILOT_DONE" in res.stdout
    assert (out_dir / "checkpoint-2" / "COMPLETE").is_file()

    train_lines = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    assert len(train_lines) == 2
    for rec in train_lines:
        assert rec["trained_blocks"] >= 2
        assert rec["kl_loss"] > 0.0
        assert rec["ce_loss"] > 0.0
