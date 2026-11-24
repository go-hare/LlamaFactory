"""Tests for Stage-2 few-step sampler distillation, progressive commit, and CLI pipeline."""
# ruff: noqa: E402

import copy
import subprocess
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from test_qwen3_vl_diffusion import make_mm_prompt, tiny_qwen3_vl

from llamafactory.train.diffusion.adapter import get_diffusion_adapter
from llamafactory.train.diffusion.loss import _get_conditioner
from llamafactory.train.diffusion.rl import (
    _commit,
    _dynamic_parallel_step,
    _frontier_commit_step,
    few_step_distill_step,
    sample_canvas,
)


def test_dynamic_parallel_step_branch_fallback_and_confident_parallel_unlock():
    vocab_size = 64
    canvas = torch.full((1, 16), 9, dtype=torch.long)
    committed = torch.zeros(1, 16, dtype=torch.bool)
    committed[0, :2] = True

    # Case 1: Branch point (uniform logits, no token meets tau=0.75) -> commits ONLY pos 2 (1 token!).
    logits_flat = torch.zeros(1, 16, vocab_size)
    next_c, next_comm, _, _ = _dynamic_parallel_step(
        logits_flat,
        canvas,
        committed,
        temperature=0.5,
        confidence_threshold=0.75,
        frontier_window=8,
        vocab_size=vocab_size,
    )
    assert torch.equal(next_c[0, :2], torch.tensor([9, 9]))
    assert bool(next_comm[0, :3].all())
    assert not bool(next_comm[0, 3:].any())

    # Case 2: High-confidence span at pos 2..5 (inside window) & pos 14 (outside window) -> unlocks 2..5 in parallel, skips 14.
    logits_sharp = torch.zeros(1, 16, vocab_size)
    logits_sharp[0, 2:6, 11] = 10.0
    logits_sharp[0, 14, 12] = 10.0
    _, next_comm2, _, _ = _dynamic_parallel_step(
        logits_sharp,
        canvas,
        committed,
        temperature=0.5,
        confidence_threshold=0.75,
        frontier_window=8,
        vocab_size=vocab_size,
    )
    assert bool(next_comm2[0, :6].all())
    assert not bool(next_comm2[0, 14])

    # Case 3: Weak 2-way tie (~50% raw prob) MUST NOT be inflated past tau=0.75 by low temperature=0.2!
    logits_tie = torch.full((1, 16, vocab_size), -20.0)
    logits_tie[0, 2:6, 11] = 0.2
    logits_tie[0, 2:6, 12] = 0.0  # raw softmax prob of 11 is ~0.55 < 0.75, even though T=0.2 gives ~0.73..0.85
    _, next_comm3, _, _ = _dynamic_parallel_step(
        logits_tie,
        canvas,
        committed,
        temperature=0.2,
        confidence_threshold=0.75,
        frontier_window=8,
        vocab_size=vocab_size,
    )
    assert bool(next_comm3[0, :3].all())
    assert not bool(next_comm3[0, 3:].any())

    # End-to-end sample_canvas(sampler="dynamic") with KV cache + multimodal Qwen3-VL works cleanly.
    model = tiny_qwen3_vl("eager")
    prompt, mm = make_mm_prompt(3)
    with torch.no_grad():
        c, ent = sample_canvas(
            model,
            prompt,
            steps=8,
            canvas_size=8,
            pad_id=0,
            sampler="dynamic",
            temperature=0.5,
            multimodal_input=mm,
            use_kv_cache=True,
            defer_adjacent_duplicates=True,
            confidence_threshold=0.75,
            frontier_window=8,
            exempt_tokens=(1,),
        )
    assert c.shape == (1, 8)
    assert torch.isfinite(ent)


def test_frontier_commit_locks_committed_and_advances_left_to_right():
    vocab_size = 64
    # Make far-right pos 14..15 slightly lower entropy than pos 2..5, but still > 0.15 so fallback must go left-to-right!
    logits = torch.zeros(1, 16, vocab_size)
    logits[0, 14:, 7] = 1.5  # entropy ~ 3.5 > 0.15
    canvas = torch.full((1, 16), 9, dtype=torch.long)
    committed = torch.zeros(1, 16, dtype=torch.bool)
    committed[0, :2] = True  # pos 0..1 already locked to token 9

    next_c, next_comm, _, _ = _frontier_commit_step(
        logits,
        canvas,
        committed,
        temperature=0.8,
        entropy_bound=0.15,
        vocab_size=vocab_size,
        target_total=6,
        is_final_step=False,
    )
    # Locked tokens 0..1 stay untouched as 9, and fallback advances strictly to pos 2..5 (NOT 14..15).
    assert torch.equal(next_c[0, :2], torch.tensor([9, 9]))
    assert bool(next_comm[0, :6].all())
    assert not bool(next_comm[0, 6:].any())

    # End-to-end sample_canvas(sampler="frontier") with KV cache + multimodal Qwen3-VL works cleanly.
    model = tiny_qwen3_vl("eager")
    prompt, mm = make_mm_prompt(3)
    with torch.no_grad():
        c, ent = sample_canvas(
            model,
            prompt,
            steps=4,
            canvas_size=8,
            pad_id=0,
            sampler="frontier",
            multimodal_input=mm,
            use_kv_cache=True,
            defer_adjacent_duplicates=True,
        )
    assert c.shape == (1, 8)
    assert torch.isfinite(ent)


def test_apply_lora_excludes_self_conditioner():
    from run_qwen3_conversion_pilot import apply_lora_if_requested

    base = tiny_qwen3_vl("eager", freeze_vision=True)
    model = apply_lora_if_requested(base, 4, freeze_vision=True)
    cond = _get_conditioner(model)
    assert not any("lora" in name for name, _ in cond.named_modules())
    assert all(p.requires_grad for p in cond.parameters())


def test_progressive_commit_guarantees_min_keep_progress():
    vocab_size = 64
    # Uniform high-entropy logits where entropy_bound=0.1 alone would commit 0-1 tokens.
    logits = torch.zeros(1, 16, vocab_size)
    logits[0, :, 5] = 0.2
    canvas = torch.randint(0, vocab_size, (1, 16))

    torch.manual_seed(0)
    out_plain, _, _ = _commit(
        logits, canvas, temperature=0.8, entropy_bound=0.1, vocab_size=vocab_size, is_final_step=False
    )
    torch.manual_seed(0)
    out_prog, _, _ = _commit(
        logits, canvas, temperature=0.8, entropy_bound=0.1, vocab_size=vocab_size, is_final_step=False, min_keep=8
    )
    # With min_keep=8, at least 8 positions commit from probs rather than uniform noise (differs cleanly).
    assert not torch.equal(out_plain, out_prog)

    # End-to-end sample_canvas with progressive_commit=True and KV cache finishes cleanly.
    model = tiny_qwen3_vl("eager")
    prompt, mm = make_mm_prompt(3)
    with torch.no_grad():
        c, ent = sample_canvas(
            model,
            prompt,
            steps=4,
            canvas_size=8,
            pad_id=0,
            sampler="entropy",
            multimodal_input=mm,
            use_kv_cache=True,
            progressive_commit=True,
            defer_adjacent_duplicates=True,
        )
    assert c.shape == (1, 8)
    assert torch.isfinite(ent)


def test_few_step_distill_step_multimodal_peft_and_frozen_teacher():
    base = tiny_qwen3_vl("eager", freeze_vision=True)
    adapter = get_diffusion_adapter(base)
    teacher_cond = copy.deepcopy(_get_conditioner(base)).eval()
    for p in teacher_cond.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=adapter.lora_target_modules(base, freeze_vision=True),
    )
    model = get_peft_model(base, lora_cfg).train()
    adapter.freeze_vision(model)
    for p in _get_conditioner(model).parameters():
        p.requires_grad_(True)

    prompt, mm = make_mm_prompt(11)
    response = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long)
    loss, metrics = few_step_distill_step(
        model,
        prompt,
        response,
        canvas_size=8,
        num_canvases=2,
        pad_id=0,
        eos_id=1,
        student_steps=4,
        teacher_substeps=2,
        train_points=2,
        loss_mode="tail-capped",
        kl_weight=0.5,
        kl_topk=16,
        teacher_conditioner=teacher_cond,
        multimodal_input=mm,
        backward_per_point=True,
        loss_scale=1.0,
    )
    assert torch.isfinite(loss)
    assert metrics["ce_loss"] > 0.0
    assert metrics["kl_loss"] >= 0.0
    # Vision encoder stays completely frozen with None grads.
    assert all(p.grad is None for p in model.get_base_model().model.visual.parameters())
    # Teacher conditioner stays untouched.
    assert all(p.grad is None for p in teacher_cond.parameters())
    # Student SelfConditioner and LoRA adapters received non-zero gradients!
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in _get_conditioner(model).parameters())
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for name, p in model.named_parameters()
        if "lora_" in name
    )


def test_stage2_fewstep_cli_tiny_vl_train_resume_and_export(tmp_path):
    out = tmp_path / "stage2_run"
    cmd = [
        sys.executable,
        str(SCRIPTS / "run_stage2_fewstep_distill.py"),
        "--tiny-vl",
        "--out",
        str(out),
        "--student-steps",
        "4",
        "--teacher-substeps",
        "2",
        "--train-points",
        "2",
        "--lora-rank",
        "4",
        "--grad-accum",
        "2",
        "--max-steps",
        "2",
        "--eval-every",
        "1",
        "--eval-steps-list",
        "2",
        "4",
        "--generation-canvases",
        "1",
        "--export-merged",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "FEWSTEP_SUMMARY step=0 S=2" in res.stdout
    assert "STAGE2_DISTILL_DONE" in res.stdout
    assert (out / "checkpoint-2/COMPLETE").read_text() == "2"
    assert (out / "merged_stage2/COMPLETE").read_text() == "merged_stage2"
    assert (out / "merged_stage2/self_conditioner.safetensors").is_file()

    # Verify resume from Stage-2 checkpoint-1 works cleanly!
    resumed = tmp_path / "stage2_resumed"
    cmd_res = [c for c in cmd if c != "--export-merged"]
    idx = cmd_res.index("--out") + 1
    cmd_res[idx] = str(resumed)
    cmd_res += ["--resume", str(out / "checkpoint-1")]
    res2 = subprocess.run(cmd_res, capture_output=True, text=True, timeout=180)
    assert res2.returncode == 0, res2.stdout + res2.stderr
    assert "STAGE2_RESUMED" in res2.stdout
    assert (resumed / "checkpoint-2/COMPLETE").read_text() == "2"


def test_load_diffusion_backbone_supports_separate_processor_name(monkeypatch):
    import transformers
    from types import SimpleNamespace

    from llamafactory.train.diffusion.adapter import load_diffusion_backbone

    seen = {}

    class DummyModel:
        def to(self, device):
            seen["device"] = device
            return self

    fake_cfg = SimpleNamespace(model_type="qwen3_vl")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(lambda cls, p: seen.setdefault("cfg", p) and fake_cfg))
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, p: seen.setdefault("processor", p) and SimpleNamespace(tokenizer="tok")),
    )
    monkeypatch.setattr(
        transformers.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        classmethod(lambda cls, p, dtype=None: seen.setdefault("model", p) and DummyModel()),
    )

    model, tok, proc, adapter = load_diffusion_backbone(
        "/tmp/local-stage1-ckpt",
        dtype=torch.bfloat16,
        device="cpu",
        processor_name="Qwen/Qwen3-VL-4B-Thinking",
    )
    assert isinstance(model, DummyModel)
    assert tok == "tok"
    assert proc.tokenizer == "tok"
    assert adapter.name == "qwen3_vl"
    assert seen == {
        "cfg": "/tmp/local-stage1-ckpt",
        "processor": "Qwen/Qwen3-VL-4B-Thinking",
        "model": "/tmp/local-stage1-ckpt",
        "device": "cpu",
    }
