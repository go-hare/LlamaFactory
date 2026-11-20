"""Unit tests for MiniCPM5-2B (LlamaForCausalLM) block-diffusion conversion and GpuMasterAdamW."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from llamafactory.train.diffusion.adapter import StandardCausalAdapter, get_diffusion_adapter
from llamafactory.train.diffusion.loss import _get_conditioner, diffusion_all_blocks_step, prepare_conversion_training
from llamafactory.train.diffusion.precision import GpuMasterAdamW
from run_qwen3_conversion_pilot import LazyQwen3Split, resolve_special_ids
from run_single_sample_controls import load_model_weights, save_checkpoint


class _FakeMiniCPM5Tokenizer:
    bos_token = "<s>"
    bos_token_id = 0
    pad_token = "</s>"
    pad_token_id = 1
    eos_token = "</s>"
    eos_token_id = [1, 130073]

    def get_vocab(self):
        return {"<s>": 0, "</s>": 1, "<|im_start|>": 130072, "<|im_end|>": 130073, "<unk>": 130074}

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) % 60 + 2 for ch in text]


class _FakeQwen3Tokenizer:
    bos_token = None
    bos_token_id = None
    pad_token = "<|endoftext|>"
    pad_token_id = 151643
    eos_token = "<|im_end|>"
    eos_token_id = 151645

    def get_vocab(self):
        return {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) % 60 + 2 for ch in text]


def _tiny_minicpm5_llama() -> LlamaForCausalLM:
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        bos_token_id=0,
        eos_token_id=[1, 120],
        pad_token_id=1,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg).to(torch.bfloat16)


def test_minicpm5_special_ids_handle_list_eos_without_pad_collision():
    tok = _FakeMiniCPM5Tokenizer()
    for fn in (resolve_special_ids, StandardCausalAdapter().resolve_special_ids):
        pad_id, eos_id, stop_tokens = fn(tok)
        assert pad_id == 1
        assert eos_id == 130073
        assert stop_tokens == (130073,)
        assert pad_id not in stop_tokens


def test_lazy_split_prepends_bos_for_minicpm5_only(tmp_path: Path):
    jsonl = tmp_path / "train.jsonl"
    row = {
        "prompt": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "response": "<think>\nok\n</think>\nhi<|im_end|>",
        "bucket": 0,
    }
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    cpm_split = LazyQwen3Split(jsonl, _FakeMiniCPM5Tokenizer())
    assert cpm_split[0]["prompt"].startswith("<s><|im_start|>user\n")

    qwen_split = LazyQwen3Split(jsonl, _FakeQwen3Tokenizer())
    assert qwen_split[0]["prompt"].startswith("<|im_start|>user\n")
    assert not qwen_split[0]["prompt"].startswith("<s>")


def test_gpu_master_adamw_exact_fp32_updates_and_resume():
    torch.manual_seed(0)
    base_w = torch.randn(32, 32, dtype=torch.bfloat16)
    p_master = torch.nn.Parameter(base_w.clone())
    p_ref = torch.nn.Parameter(base_w.float().clone())

    opt_master = GpuMasterAdamW([p_master], lr=1e-5, betas=(0.95, 0.99), weight_decay=1e-4)
    opt_ref = torch.optim.AdamW([p_ref], lr=1e-5, betas=(0.95, 0.99), weight_decay=1e-4)

    for step in range(10):
        g = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
        p_master.grad = g.clone()
        p_ref.grad = g.float().clone()
        opt_master.step()
        opt_ref.step()

    # Master FP32 tensor matches pure FP32 AdamW to numerical precision.
    master_fp32 = opt_master.state[p_master]["master"]
    assert (master_fp32 - p_ref.detach()).abs().max().item() < 1e-6
    assert (p_master.detach().float() - p_ref.detach().to(torch.bfloat16).float()).abs().max().item() == 0.0

    # Roundtrip state_dict reload preserves exact continuation.
    saved = opt_master.state_dict()
    p_reloaded = torch.nn.Parameter(p_master.detach().clone())
    opt_reloaded = GpuMasterAdamW([p_reloaded], lr=1e-5, betas=(0.95, 0.99), weight_decay=1e-4)
    opt_reloaded.load_state_dict(saved)

    g_next = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
    p_master.grad = g_next.clone()
    p_reloaded.grad = g_next.clone()
    opt_master.step()
    opt_reloaded.step()
    assert torch.equal(opt_master.state[p_master]["master"], opt_reloaded.state[p_reloaded]["master"])


def test_minicpm5_full_param_all_blocks_train_save_and_resume(tmp_path: Path):
    model = _tiny_minicpm5_llama()
    assert isinstance(get_diffusion_adapter(model), StandardCausalAdapter)
    prepare_conversion_training(model)
    model.train()

    optimizer = GpuMasterAdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    prompts = [torch.tensor([0, 10, 11, 12]), torch.tensor([0, 13, 14, 15, 16])]
    responses = [torch.tensor([20, 21, 22, 23, 120]), torch.tensor(list(range(30, 50)) + [120])]

    torch.manual_seed(7)
    loss, parts = diffusion_all_blocks_step(
        model,
        prompts,
        responses,
        canvas_size=16,
        num_canvases=4,
        pad_id=1,
        eos_id=120,
        diagnostics=True,
        max_train_blocks=4,
    )
    assert torch.isfinite(loss)
    assert parts["trained_blocks"] == 3
    loss.backward()
    optimizer.step()

    out_dir = tmp_path / "minicpm5_run"
    out_dir.mkdir()
    save_checkpoint(model, None, optimizer, out_dir, {"global_step": 1, "settings": {"model": "openbmb/MiniCPM5-2B"}})
    ckpt = out_dir / "checkpoint-1"
    assert (ckpt / "COMPLETE").is_file()

    # Reload into a fresh model + fresh GpuMasterAdamW and verify exact weights & state parity.
    model2 = _tiny_minicpm5_llama()
    prepare_conversion_training(model2)
    load_model_weights(model2, ckpt)
    from safetensors.torch import load_file

    _get_conditioner(model2).load_state_dict(load_file(str(ckpt / "self_conditioner.safetensors")), strict=True)
    opt2 = GpuMasterAdamW([p for p in model2.parameters() if p.requires_grad], lr=1e-4)
    restored = torch.load(ckpt / "diagnostic_state.pt", map_location="cpu", weights_only=False)
    opt2.load_state_dict(restored["optimizer"])

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1.detach(), p2.detach()), f"Weight mismatch after resume: {n1}"


def test_minicpm5_llama_canvas_is_truly_bidirectional():
    """Mutating the last token in a MiniCPM5/Llama canvas must change the first canvas token's logits."""
    from llamafactory.train.diffusion.loss import denoise_step_logits

    model = _tiny_minicpm5_llama().to(torch.float32)
    prepare_conversion_training(model)
    model.eval()

    prompt = torch.tensor([0, 10, 11, 12], dtype=torch.long)
    canvas_a = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27], dtype=torch.long)
    canvas_b = canvas_a.clone()
    canvas_b[-1] = 99
    zero_sc = torch.zeros(1, 8, 32, dtype=torch.float32)

    with torch.no_grad():
        logits_a = denoise_step_logits(model, prompt, canvas_a, sc_embeddings=zero_sc, pad_id=1)
        logits_b = denoise_step_logits(model, prompt, canvas_b, sc_embeddings=zero_sc, pad_id=1)
    # Position 0 in canvas must see position 7 in canvas (bidirectional, NOT overwritten to causal!).
    assert (logits_a[:, 0] - logits_b[:, 0]).abs().max().item() > 1e-4
