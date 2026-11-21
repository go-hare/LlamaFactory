"""Tests for Qwen3-VL multimodal block-diffusion, DiffusionHistoryCache, and entropy sampler polish."""
# ruff: noqa: E402

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3VLConfig, Qwen3VLForConditionalGeneration


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from llamafactory.train.diffusion.adapter import Qwen3VLAdapter, get_diffusion_adapter
from llamafactory.train.diffusion.loss import (
    DiffusionHistoryCache,
    _get_conditioner,
    denoise_step_logits,
    diffusion_all_blocks_step,
    diffusion_batch_step,
    prepare_conversion_training,
    supports_history_kv_cache,
)
from llamafactory.train.diffusion.objective import CanvasMeanObjective, TailCappedObjective, resolve_objective
from llamafactory.train.diffusion.rl import _apply_local_repetition_penalty, _commit, sample_canvas, sample_response


IMG_TOK = 60
VID_TOK = 61
VIS_START = 58
VIS_END = 59
PATCH_DIM = 3 * 2 * 4 * 4  # in_channels * temporal_patch_size * patch_size^2


def tiny_qwen3_vl(attn: str = "eager", *, freeze_vision: bool = True) -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(23)
    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default"},
        ),
        vision_config=dict(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=16,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        tie_word_embeddings=True,
        image_token_id=IMG_TOK,
        video_token_id=VID_TOK,
        vision_start_token_id=VIS_START,
        vision_end_token_id=VIS_END,
    )
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    model.set_attn_implementation(attn)
    prepare_conversion_training(model)
    if freeze_vision:
        get_diffusion_adapter(model).freeze_vision(model)
    return model


def make_mm_prompt(seed: int = 0) -> tuple[torch.Tensor, dict]:
    # grid [1, 2, 2] with spatial_merge_size=2 -> 1 merged vision token, 4 raw patches
    g = torch.Generator().manual_seed(seed)
    pv = torch.randn(4, PATCH_DIM, generator=g)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    prompt = torch.tensor([2, VIS_START, IMG_TOK, VIS_END, 5, 6], dtype=torch.long)
    return prompt, {"pixel_values": pv, "image_grid_thw": grid}


def test_qwen3_vl_adapter_registration_freeze_and_lora_isolation():
    model = tiny_qwen3_vl(freeze_vision=False)
    adapter = get_diffusion_adapter(model)
    assert isinstance(adapter, Qwen3VLAdapter)
    assert adapter.embed_scale(model) == 1.0
    assert supports_history_kv_cache(model) is True

    frozen = adapter.freeze_vision(model)
    assert frozen > 0
    assert all(not p.requires_grad for p in model.model.visual.parameters())
    assert any(p.requires_grad for p in model.model.language_model.parameters())

    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=adapter.lora_target_modules(model, freeze_vision=True),
    )
    peft_model = get_peft_model(model, lora_cfg)
    for p in _get_conditioner(peft_model).parameters():
        p.requires_grad_(True)
    # Zero LoRA or trainable params inside visual encoder.
    for name, param in peft_model.named_parameters():
        if ".visual." in name:
            assert not param.requires_grad, f"Visual param unexpectedly trainable: {name}"
    # Mixed vision + text backward through PEFT Qwen3-VL works cleanly.
    prompt_img, mm = make_mm_prompt(1)
    prompt_txt = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    resp = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    loss, _ = diffusion_all_blocks_step(
        peft_model.train(),
        [prompt_img, prompt_txt],
        [resp, resp],
        canvas_size=4,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
        multimodal_inputs=[mm, None],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in peft_model.get_base_model().model.visual.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in _get_conditioner(peft_model).parameters())


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_qwen3_vl_multimodal_bidirectionality_and_noise_collision_guard(attn):
    model = tiny_qwen3_vl(attn)
    prompt, mm_a = make_mm_prompt(10)
    _, mm_b = make_mm_prompt(99)
    # Deliberately place IMG_TOK and VID_TOK inside the noisy canvas to test placeholder guard!
    canvas_a = torch.tensor([IMG_TOK, 7, VID_TOK, 9], dtype=torch.long)
    canvas_b = torch.tensor([IMG_TOK, 7, VID_TOK, 33], dtype=torch.long)
    zero = torch.zeros(1, 4, 16)

    with torch.no_grad():
        logits_a = denoise_step_logits(model, prompt, canvas_a, sc_embeddings=zero, pad_id=0, multimodal_input=mm_a)
        logits_img_changed = denoise_step_logits(model, prompt, canvas_a, sc_embeddings=zero, pad_id=0, multimodal_input=mm_b)
        logits_last_changed = denoise_step_logits(model, prompt, canvas_b, sc_embeddings=zero, pad_id=0, multimodal_input=mm_a)

    # Image pixels genuinely affect canvas logits.
    assert float((logits_a - logits_img_changed).abs().max()) > 1e-4
    # Mutating the LAST canvas token changes the FIRST canvas token -> truly bidirectional!
    assert float((logits_a[:, 0] - logits_last_changed[:, 0]).abs().max()) > 1e-5


def test_qwen3_vl_all_blocks_exact_parity_with_single_block_and_mixed_batch():
    model = tiny_qwen3_vl("eager")
    prompt_img, mm = make_mm_prompt(5)
    response = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long)
    canvas_size = 4
    noisy_blk0 = torch.tensor([IMG_TOK, 21, 22, 23], dtype=torch.long)
    noisy_blk1 = torch.tensor([24, 25, VID_TOK, 27], dtype=torch.long)
    zero = torch.zeros(1, canvas_size, 16)

    with torch.no_grad():
        ref_blk0 = denoise_step_logits(model, prompt_img, noisy_blk0, sc_embeddings=zero, pad_id=0, multimodal_input=mm)
        hist1 = torch.cat([prompt_img, response[:canvas_size]], dim=0)
        ref_blk1 = denoise_step_logits(model, hist1, noisy_blk1, sc_embeddings=zero, pad_id=0, multimodal_input=mm)

    # Run all-blocks with deterministic corruption matching noisy_blk0 / noisy_blk1 and zero self-conditioning.
    from llamafactory.train.diffusion.objective import CorruptedSample

    canned = [
        CorruptedSample(noisy=noisy_blk0, corrupted=torch.ones(canvas_size, dtype=torch.bool), time=0.5),
        CorruptedSample(noisy=noisy_blk1, corrupted=torch.ones(canvas_size, dtype=torch.bool), time=0.5),
    ]
    it = iter(canned)
    captured = {}

    orig_fwd = __import__("llamafactory.train.diffusion.loss", fromlist=["_diffusion_forward"])._diffusion_forward

    def spy_fwd(*args, **kwargs):
        out = orig_fwd(*args, **kwargs)
        captured["logits"] = out.logits.detach().clone()
        return out

    with (
        torch.no_grad(),
        patch("llamafactory.train.diffusion.objective.UniformNoiseCorruption.corrupt", side_effect=lambda *a, **kw: next(it)),
        patch("llamafactory.train.diffusion.loss._diffusion_forward", side_effect=spy_fwd),
        patch("torch.rand", return_value=torch.ones((1, 1, 1))),  # disable Pass-1 SC
    ):
        loss, parts = diffusion_all_blocks_step(
            model,
            [prompt_img],
            [response],
            canvas_size=canvas_size,
            num_canvases=2,
            pad_id=0,
            eos_id=1,
            multimodal_inputs=[mm],
        )
    assert parts["trained_blocks"] == 2
    assert torch.isfinite(loss)
    torch.testing.assert_close(captured["logits"][:, :canvas_size], ref_blk0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(captured["logits"][:, canvas_size:], ref_blk1, atol=1e-4, rtol=1e-4)

    # Mixed batch (row 0 multimodal + row 1 text-only) full backward test with frozen ViT.
    model.train()
    get_diffusion_adapter(model).freeze_vision(model)
    loss_mix, _ = diffusion_batch_step(
        model,
        [prompt_img, torch.tensor([2, 3, 4], dtype=torch.long)],
        [response, torch.tensor([12, 13, 14, 15], dtype=torch.long)],
        canvas_size=canvas_size,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
        multimodal_inputs=[mm, None],
    )
    loss_mix.backward()
    assert torch.isfinite(loss_mix)
    assert all(p.grad is None for p in model.model.visual.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.model.language_model.parameters())


@pytest.mark.parametrize("kind", ["qwen3_vl_mm", "qwen3", "llama"])
def test_history_kv_cache_exact_parity_across_steps_and_blocks(kind):
    torch.manual_seed(77)
    if kind == "qwen3_vl_mm":
        model = tiny_qwen3_vl("eager")
        prompt, mm = make_mm_prompt(42)
    elif kind == "qwen3":
        cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        model = Qwen3ForCausalLM(cfg).eval()
        prepare_conversion_training(model)
        prompt, mm = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long), None
    else:
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        model = LlamaForCausalLM(cfg).eval()
        prepare_conversion_training(model)
        prompt, mm = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long), None

    canvas0_a = torch.tensor([10, 11, 12, 13], dtype=torch.long)
    canvas0_b = torch.tensor([14, 15, 16, 17], dtype=torch.long)
    committed_blk0 = torch.tensor([20, 21, 22, 23], dtype=torch.long)
    canvas1 = torch.tensor([30, 31, 32, 33], dtype=torch.long)
    sc_a = torch.randn(1, 4, 16) * 0.1
    sc_b = torch.randn(1, 4, 16) * 0.1

    cache = DiffusionHistoryCache(model, pad_id=0)
    cache.prefill(prompt, multimodal_input=mm)
    assert cache.seq_length == prompt.numel()

    with torch.no_grad():
        c_step1 = cache.denoise_step_logits(canvas0_a, sc_embeddings=sc_a)
        u_step1 = denoise_step_logits(model, prompt, canvas0_a, sc_embeddings=sc_a, pad_id=0, multimodal_input=mm)
        torch.testing.assert_close(c_step1, u_step1, atol=1e-4, rtol=1e-4)
        assert cache.seq_length == prompt.numel()  # cropped cleanly!

        c_step2 = cache.denoise_step_logits(canvas0_b, sc_embeddings=sc_b)
        u_step2 = denoise_step_logits(model, prompt, canvas0_b, sc_embeddings=sc_b, pad_id=0, multimodal_input=mm)
        torch.testing.assert_close(c_step2, u_step2, atol=1e-4, rtol=1e-4)
        assert cache.seq_length == prompt.numel()

        # Append committed block 0 and verify block 1 matches uncached full history!
        cache.prefill(committed_blk0)
        assert cache.seq_length == prompt.numel() + committed_blk0.numel()
        full_hist1 = torch.cat([prompt, committed_blk0], dim=0)
        c_blk1 = cache.denoise_step_logits(canvas1, sc_embeddings=sc_a)
        u_blk1 = denoise_step_logits(model, full_hist1, canvas1, sc_embeddings=sc_a, pad_id=0, multimodal_input=mm)
        torch.testing.assert_close(c_blk1, u_blk1, atol=1e-4, rtol=1e-4)

        # End-to-end sample_response with vs without KV cache produces identical tokens under same seed!
        torch.manual_seed(1234)
        out_cached = sample_response(
            model, prompt, max_canvases=2, steps=3, canvas_size=4, pad_id=0, stop_tokens=(1,),
            sampler="entropy", multimodal_input=mm, use_kv_cache=True,
        )
        torch.manual_seed(1234)
        out_uncached = sample_response(
            model, prompt, max_canvases=2, steps=3, canvas_size=4, pad_id=0, stop_tokens=(1,),
            sampler="entropy", multimodal_input=mm, use_kv_cache=False,
        )
        assert torch.equal(out_cached, out_uncached)


def test_entropy_local_repetition_penalty_and_temperature_scaling():
    logits = torch.zeros(1, 6, 16)
    logits[:, :, 5] = 4.0  # repeated comma-like token
    logits[:, :, 1] = 4.0  # exempt stop token
    canvas_comma = torch.tensor([[5, 5, 5, 5, 5, 5]], dtype=torch.long)
    penalized = _apply_local_repetition_penalty(logits, canvas_comma, penalty=2.0, exempt_tokens=(1,))
    # Non-stop repeated token logit 4.0 -> 2.0 at every position with >=2 matching neighbors.
    assert float(penalized[0, 2, 5]) == pytest.approx(2.0)
    assert float(penalized[0, 2, 1]) == pytest.approx(4.0)

    # Exempt stop-token tail is never penalized.
    canvas_eos = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long)
    penalized_eos = _apply_local_repetition_penalty(logits, canvas_eos, penalty=2.0, exempt_tokens=(1,))
    torch.testing.assert_close(penalized_eos, logits)

    # Temperature argument scales entropy schedule without breaking default temperature=0.7.
    model = tiny_qwen3_vl("eager")
    seen_temps = []
    from llamafactory.train.diffusion import rl as rl_mod

    orig_commit = rl_mod._commit

    def spy_commit(*args, **kwargs):
        seen_temps.append(kwargs["temperature"])
        return orig_commit(*args, **kwargs)

    with torch.no_grad(), patch.object(rl_mod, "_commit", side_effect=spy_commit):
        sample_canvas(model, torch.tensor([2, 3]), steps=2, canvas_size=4, vocab_size=64, sampler="entropy", temperature=0.35, early_stop=False)
    assert seen_temps[0] == pytest.approx(0.4)


def test_tail_capped_objective_balances_short_vqa_without_amplifying_short_tails():
    obj_cap = resolve_objective("tail-capped")
    assert isinstance(obj_cap, TailCappedObjective)
    obj_mean = CanvasMeanObjective()

    # Case 1: Long text block (240 body, 16 tail) -> 0.25*240/16 = 3.75 >= 1.0 -> IDENTICAL to canvas-mean!
    torch.manual_seed(0)
    logits = torch.randn(256, 32)
    targets = torch.randint(0, 32, (256,))
    valid = torch.ones(256, dtype=torch.bool)
    corrupted = torch.ones(256, dtype=torch.bool)
    loss_cap = obj_cap.compute(logits, targets, valid, corrupted, body_tokens=240, time=0.5)
    loss_mean = obj_mean.compute(logits, targets, valid, corrupted, body_tokens=240, time=0.5)
    torch.testing.assert_close(loss_cap, loss_mean, atol=1e-6, rtol=1e-6)

    # Case 2: Short vision VQA block (32 body, 224 easy EOS tail) -> body keeps 80% total weight, not 12.5%!
    logits_v = torch.zeros(256, 32)
    targets_v = torch.ones(256, dtype=torch.long)
    # Tail positions predict target with near-zero CE; body positions have uniform CE = log(32) ~= 3.4657.
    logits_v[32:, 1] = 30.0
    loss_v_cap = obj_cap.compute(logits_v, targets_v, valid, corrupted, body_tokens=32, time=0.5)
    loss_v_mean = obj_mean.compute(logits_v, targets_v, valid, corrupted, body_tokens=32, time=0.5)
    body_ce = float(torch.log(torch.tensor(32.0)))
    assert float(loss_v_mean) == pytest.approx(body_ce * (32.0 / 256.0), rel=1e-3)
    # Tail total weight capped to 0.25 * 32 = 8 -> body share = 32 / (32 + 8) = 0.80!
    assert float(loss_v_cap) == pytest.approx(body_ce * 0.80, rel=1e-3)


def test_entropy_defer_adjacent_duplicates_breaks_stutter_and_keeps_true_reduplication():
    vocab_size = 64
    logits = torch.full((1, 4, vocab_size), -20.0)
    logits[0, 0, 5] = 20.0
    logits[0, 1, 5] = 20.0
    logits[0, 2, 3] = 20.0
    logits[0, 3, 1] = 20.0

    # Step 1: pos 0 newly changes (9 -> 5) and pos 1 simultaneously predicts 5 -> pos 1 is deferred!
    torch.manual_seed(7)
    canvas_noise = torch.tensor([[9, 8, 7, 6]], dtype=torch.long)
    out1, _, _ = _commit(
        logits,
        canvas_noise,
        temperature=0.5,
        entropy_bound=0.1,
        vocab_size=vocab_size,
        is_final_step=False,
        defer_adjacent_duplicates=True,
        exempt_tokens=(1,),
    )
    assert int(out1[0, 0]) == 5
    assert int(out1[0, 2]) == 3
    assert int(out1[0, 3]) == 1
    assert int(out1[0, 1]) != 5  # deferred to random noise on this step!

    # Step 2: pos 0 is ALREADY settled as 5 -> if model still predicts [5, 5] (true reduplication), pos 1 commits!
    out2, _, _ = _commit(
        logits,
        out1,
        temperature=0.5,
        entropy_bound=0.1,
        vocab_size=vocab_size,
        is_final_step=False,
        defer_adjacent_duplicates=True,
        exempt_tokens=(1,),
    )
    assert out2[0].tolist() == [5, 5, 3, 1]

    # Exempt EOS tail [1, 1, 1, 1] is never deferred even on step 1.
    eos_logits = torch.full((1, 4, vocab_size), -20.0)
    eos_logits[:, :, 1] = 20.0
    out_eos, _, _ = _commit(
        eos_logits,
        canvas_noise,
        temperature=0.5,
        entropy_bound=0.1,
        vocab_size=vocab_size,
        is_final_step=False,
        defer_adjacent_duplicates=True,
        exempt_tokens=(1,),
    )
    assert out_eos[0].tolist() == [1, 1, 1, 1]


def test_qwen3_vl_pilot_cli_tiny_vl_all_blocks_and_resume(tmp_path):
    out = tmp_path / "qwen3_vl_full"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_qwen3_conversion_pilot.py"),
        "--tiny-vl",
        "--training-mode",
        "all-blocks",
        "--loss-mode",
        "tail-capped",
        "--lora-rank",
        "0",
        "--optimizer-mode",
        "gpu-fp32-master",
        "--sampler",
        "entropy",
        "--repetition-penalty",
        "1.15",
        "--defer-adjacent-dup",
        "--eval-canvas",
        "8",
        "--vision-ratio",
        "0.5",
        "--out",
        str(out),
        "--max-steps",
        "2",
        "--eval-every",
        "1",
        "--generation-every",
        "2",
        "--micro-batch",
        "2",
        "--grad-accum",
        "1",
        "--sample-steps",
        "2",
        "--generation-canvases",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "VISION_ENCODER_FROZEN" in res.stdout
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"
    assert (out / "eval-0002.json").is_file()

    # Resume from checkpoint-1 and verify tied Qwen3-VL full-param resume works cleanly!
    resumed = tmp_path / "qwen3_vl_resumed"
    cmd_res = cmd[: cmd.index("--out") + 1] + [str(resumed)] + cmd[cmd.index("--out") + 2 :] + ["--resume", str(out / "checkpoint-1")]
    res2 = subprocess.run(cmd_res, capture_output=True, text=True, timeout=180)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert (resumed / "checkpoint-2/COMPLETE").read_text() == "2"
