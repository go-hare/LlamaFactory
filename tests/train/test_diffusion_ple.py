"""Regression tests for the official DiffusionGemma PLE preprocessing order."""
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from llamafactory.train.diffusion.loss import (
    _diffusion_forward, _get_conditioner, _gemma_masks, diffusion_batch_step,
    denoise_step_logits,
)


def tiny_model():
    torch.manual_seed(7)
    cfg = Gemma4TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, hidden_size_per_layer_input=4, sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg)).eval()
    _get_conditioner(model)
    return model


def test_ple_order_identity_and_gradients():
    model = tiny_model()
    text = model.model.language_model
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]])
    start = 3
    raw = text.embed_tokens(tokens)
    identity = text.get_per_layer_inputs(tokens, raw)
    expected = torch.cat([
        text.project_per_layer_inputs(raw[:, :start], identity[:, :start]),
        text.project_per_layer_inputs(raw[:, start:], None),
    ], dim=1)
    captured = []
    handle = text.layers[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[1]),
    )
    valid = torch.ones_like(tokens, dtype=torch.bool)
    allowed = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    signal = torch.randn(1, 3, 16, requires_grad=True)
    for sc in (torch.zeros_like(signal), signal):
        _diffusion_forward(model, tokens, sc, start, _gemma_masks(allowed, torch.float32),
                           valid.long().cumsum(-1) - 1)
        # Layer 0 receives its slice of the precomputed PLE.
        torch.testing.assert_close(captured[-1], expected[:, :, 0])
    handle.remove()
    ple = captured[-1]
    # Canvas PLE must not depend on SC or the identity table.
    signal_grad = torch.autograd.grad(ple.sum(), signal, allow_unused=True, retain_graph=True)[0]
    assert signal_grad is None
    table = text.embed_tokens_per_layer.weight
    canvas_grad = torch.autograd.grad(ple[:, start:].square().sum(), table,
                                     allow_unused=True, retain_graph=True)[0]
    assert canvas_grad is None or torch.count_nonzero(canvas_grad) == 0
    history_grad = torch.autograd.grad(ple[:, :start].square().sum(), table)[0]
    assert torch.count_nonzero(history_grad) > 0
    assert "project_per_layer_inputs" not in vars(text)
    assert not text.embed_tokens._forward_hooks


def test_training_backward_and_sampling():
    model = tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prompts = [torch.tensor([2, 3]), torch.tensor([2, 8, 9])]
    responses = [torch.tensor([10, 11, 12]), torch.tensor([13, 14, 15, 16, 17])]
    loss, _ = diffusion_batch_step(model, prompts, responses, canvas_size=4,
                                  num_canvases=2, pad_id=0, eos_id=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.language_model.per_layer_model_projection.weight.grad is not None
    model.eval()
    with torch.no_grad():
        logits = denoise_step_logits(model, prompts[0], responses[0],
                                    sc_embeddings=torch.zeros(1, 3, 16), pad_id=0)
    assert logits.shape == (1, 3, 64)
    assert torch.isfinite(logits).all()


def test_roundtrip_and_causal_history(tmp_path):
    from safetensors.torch import save_file, load_file
    model = tiny_model()
    tokens = torch.tensor([[2, 3, 4, 5, 6]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    from llamafactory.train.diffusion.loss import _allow_mask
    masks = _gemma_masks(_allow_mask(valid[:, :3], valid[:, 3:], canvas_bidirectional=True), torch.float32)
    signal = torch.ones(1, 2, 16)
    positions = torch.arange(5)[None]
    with torch.no_grad():
        original = _diffusion_forward(model, tokens, signal, 3, masks, positions).logits
        altered = tokens.clone(); altered[:, 3:] = torch.tensor([20, 21])
        other = _diffusion_forward(model, altered, signal, 3, masks, positions).logits
    torch.testing.assert_close(original[:, :3], other[:, :3])
    save_file({k: v.detach().clone() for k,v in model.state_dict().items()}, str(tmp_path/'model.safetensors'))
    fresh = tiny_model()
    fresh.load_state_dict(load_file(str(tmp_path/'model.safetensors')), strict=True)
    with torch.no_grad():
        loaded = _diffusion_forward(fresh, tokens, signal, 3, masks, positions).logits
    torch.testing.assert_close(original, loaded, rtol=0, atol=0)


def test_optimizer_precision_is_measured():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([param], lr=1.125e-4)
    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert optimizer.state[param]['exp_avg'].dtype == torch.bfloat16
    assert torch.equal(param, before)  # this update rounds away at magnitude 1


def test_cleanup_on_failure():
    model = tiny_model()
    text = model.model.language_model
    try:
        _diffusion_forward(model, torch.tensor([[2, 3, 4]]), torch.ones(1, 1, 17),
                           2, {}, torch.arange(3)[None])
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Expected a mismatched SC shape to fail")
    assert "project_per_layer_inputs" not in vars(text)
    assert not text.embed_tokens._forward_hooks
