"""Tests for the official Hackable Diffusion DiscreteDDIMStep sampler in rl.py."""

import pytest
import torch
from test_diffusion_ple import tiny_model

from llamafactory.train.diffusion.rl import _ddim_step, sample_canvas, sample_response


def test_ddim_final_step_has_zero_noise_and_commits_clean_tokens():
    torch.manual_seed(123)
    vocab_size = 64
    canvas = torch.tensor([[10, 20, 30, 40]])
    target = torch.tensor([[5, 20, 15, 1]])
    logits = torch.full((1, 4, vocab_size), -1e4)
    logits.scatter_(2, target.unsqueeze(-1), 1e4)

    next_canvas, shaped, entropy = _ddim_step(
        logits,
        canvas,
        time_now=0.05,
        time_next=0.0,
        temperature=0.7,
        vocab_size=vocab_size,
    )
    assert torch.equal(next_canvas, target)
    assert shaped.shape == (1, 4, vocab_size)
    assert torch.all(entropy >= 0)


def test_ddim_preserves_matching_tokens_via_bayesian_t1_posterior():
    torch.manual_seed(7)
    canvas_small = torch.full((1, 256), 17, dtype=torch.long)
    logits_full = torch.full((1, 256, 64), -1e4)
    logits_full[:, :, 17] = 1e4

    next_canvas, _, _ = _ddim_step(
        logits_full,
        canvas_small,
        time_now=0.5,
        time_next=0.4,
        temperature=0.7,
        vocab_size=64,
    )
    # With T1 = r * alpha_s = (0.5/0.6)*0.6 = 0.5 vs p_noise = (1 - 5/6)*0.4*(1/64) = 0.00104,
    # >99% of already-matching tokens stay at 17 instead of being wiped back to random noise.
    assert (next_canvas == 17).float().mean() > 0.95


def test_ddim_sample_canvas_and_response_propagate_self_conditioning():
    model = tiny_model()
    model.eval()
    prompt = torch.tensor([2, 3, 4])
    seen_sc = []
    from unittest.mock import patch
    import llamafactory.train.diffusion.rl as rl_mod

    orig_denoise = rl_mod.denoise_step_logits

    def spy_denoise(model, p, c, *, sc_embeddings, pad_id):
        seen_sc.append(sc_embeddings.detach().clone())
        return orig_denoise(model, p, c, sc_embeddings=sc_embeddings, pad_id=pad_id)

    torch.manual_seed(99)
    with torch.no_grad(), patch.object(rl_mod, "denoise_step_logits", side_effect=spy_denoise):
        canvas, ent = sample_canvas(
            model,
            prompt,
            steps=4,
            canvas_size=8,
            vocab_size=64,
            early_stop=False,
            sampler="ddim",
            temperature=0.7,
        )
    assert canvas.shape == (1, 8)
    assert len(seen_sc) == 4
    assert torch.count_nonzero(seen_sc[0]) == 0
    assert all(torch.count_nonzero(s) > 0 for s in seen_sc[1:])
    assert torch.isfinite(ent)

    torch.manual_seed(99)
    with torch.no_grad():
        out = sample_response(
            model,
            prompt,
            max_canvases=2,
            steps=3,
            canvas_size=8,
            vocab_size=64,
            sampler="ddim",
            temperature=0.7,
        )
    assert 1 <= out.numel() <= 16


def test_unknown_sampler_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="Unknown diffusion sampler"):
        sample_canvas(model, torch.tensor([2, 3]), steps=2, canvas_size=4, vocab_size=64, sampler="invalid")
