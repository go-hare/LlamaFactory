"""Official-source alignment regressions for noise IDs, init and stopping."""
from unittest.mock import patch
import pytest
import torch
from test_diffusion_ple import tiny_model
from llamafactory.train.diffusion.loss import SelfConditioner, _diffusion_forward, _gemma_masks, prepare_conversion_training
from llamafactory.train.diffusion.rl import sample_canvas


def test_noise_special_ids_are_not_replaced_and_cleanup():
    model=tiny_model()
    model.config.image_token_id=10
    model.config.audio_token_id=11
    model.config.video_token_id=12
    tokens=torch.tensor([[2,3,10,11,12,5]])
    seen=[]
    text=model.model.language_model
    h=text.embed_tokens.register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
    mask=_gemma_masks(torch.ones(1,1,6,6,dtype=torch.bool),torch.float32)
    with torch.no_grad():
        _diffusion_forward(model,tokens,torch.zeros(1,3,16),3,mask,torch.arange(6)[None])
    torch.testing.assert_close(seen[0],tokens)
    assert 'get_placeholder_mask' not in vars(model.model)
    # After the temporary text-only call, regular multimodal routing remains.
    seen.clear()
    with torch.no_grad():model(input_ids=tokens,use_cache=False)
    torch.testing.assert_close(seen[0],torch.tensor([[2,3,0,0,0,5]]))
    h.remove()


def test_special_routing_restored_on_exception():
    model=tiny_model()
    try:
        _diffusion_forward(model,torch.tensor([[2,3,4]]),torch.ones(1,1,17),2,{},torch.arange(3)[None])
    except RuntimeError:pass
    else:raise AssertionError('expected invalid signal to fail')
    assert 'get_placeholder_mask' not in vars(model.model)


def test_conditioner_official_normal_initialization():
    torch.manual_seed(5)
    cond=SelfConditioner(256,512)
    for layer in (cond.gate_proj,cond.up_proj,cond.down_proj):
        assert abs(float(layer.weight.detach().mean()))<.0002
        assert abs(float(layer.weight.detach().std())-.01)<.0002
    torch.testing.assert_close(cond.pre_norm.weight,torch.ones(256))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_self_conditioning_preserves_zero_parameter_gradients(dtype):
    torch.manual_seed(42)
    conditioner = SelfConditioner(16, 32).to(dtype)
    canvas = torch.randn(1, 4, 16, dtype=dtype, requires_grad=True)
    output = conditioner(canvas, torch.zeros_like(canvas))
    torch.testing.assert_close(output, conditioner.post_norm(canvas), rtol=0, atol=0)
    (output * torch.randn_like(output)).sum().backward()
    for parameter in conditioner.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    assert canvas.grad is not None and torch.isfinite(canvas.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_four_sc_off_microsteps_continue_adamw_momentum_and_step(dtype):
    torch.manual_seed(42)
    conditioner = SelfConditioner(16, 32).to(dtype)
    parameters = list(conditioner.parameters())
    expected = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    kwargs = dict(lr=1.125e-4, betas=(0.95, 0.99), weight_decay=1e-4, foreach=False)
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    reference = torch.optim.AdamW(expected, **kwargs)
    canvas = torch.randn(1, 4, 16, dtype=dtype, requires_grad=True)
    signal = torch.randn_like(canvas)
    weights = torch.randn_like(canvas)
    (conditioner(canvas, signal) * weights).sum().backward()
    for parameter, baseline in zip(parameters, expected):
        baseline.grad = parameter.grad.detach().clone()
    optimizer.step()
    reference.step()
    optimizer.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in parameters]
    for _ in range(4):
        (conditioner(canvas, torch.zeros_like(signal)) * weights).sum().div(4).backward()
    for parameter, baseline in zip(parameters, expected):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
        baseline.grad = torch.zeros_like(baseline)
    optimizer.step()
    reference.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters))
    for parameter, baseline in zip(parameters, expected):
        torch.testing.assert_close(parameter, baseline, rtol=0, atol=0)
        actual_state, expected_state = optimizer.state[parameter], reference.state[baseline]
        assert actual_state["step"].item() == expected_state["step"].item() == 2
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)


def test_noisy_pad_keeps_input_embedding_gradient_and_restores_padding():
    from llamafactory.train.diffusion.loss import denoise_step_logits

    model = tiny_model()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    embedding = model.model.language_model.embed_tokens
    assert embedding.padding_idx == 0
    logits = denoise_step_logits(model, torch.tensor([2, 3]), torch.tensor([0, 4, 5, 6]),
                                sc_embeddings=torch.zeros(1, 4, 16), pad_id=0)
    logits[0, 0, 7].backward()
    assert embedding.weight.grad[0].abs().sum() > 0
    assert embedding.padding_idx == 0
    with pytest.raises(RuntimeError):
        _diffusion_forward(model, torch.tensor([[2, 0]]), torch.ones(1, 1, 17), 1, {}, torch.arange(2)[None])
    assert embedding.padding_idx == 0


def test_conversion_layer_scales_are_trainable_idempotent_and_saved(tmp_path):
    from safetensors import safe_open

    model = tiny_model()
    layer = model.model.language_model.layers[0]
    old = layer.layer_scalar.detach().clone()
    assert "layer_scalar" in dict(layer.named_buffers())
    prepare_conversion_training(model)
    scalar = layer.layer_scalar
    assert isinstance(scalar, torch.nn.Parameter) and scalar.requires_grad
    assert "layer_scalar" not in dict(layer.named_buffers())
    torch.testing.assert_close(scalar, old, rtol=0, atol=0)
    prepare_conversion_training(model)
    assert layer.layer_scalar is scalar
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.125e-4, weight_decay=1e-4)
    scalar.sum().backward()
    optimizer.step()
    assert not torch.equal(scalar, old)
    assert optimizer.state[scalar]["step"] == 1
    model.save_pretrained(tmp_path)
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as handle:
        assert "model.language_model.layers.0.layer_scalar" in handle.keys()
    restored = tiny_model()
    prepare_conversion_training(restored)
    from safetensors.torch import load_file
    state = load_file(str(tmp_path / "model.safetensors"))
    for name, parameter in restored.named_parameters(remove_duplicate=False):
        if parameter is restored.lm_head.weight and name not in state:
            state[name] = state["model.language_model.embed_tokens.weight"]
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored.model.language_model.layers[0].layer_scalar, scalar, rtol=0, atol=0)


def test_early_stop_retains_sampled_tokens():
    model=tiny_model()
    torch.manual_seed(42)
    initial=torch.randint(0,64,(1,4))
    # Deliberately make sampled tokens differ from argmax despite meeting stop
    # conditions; this makes the returned-state branch observable.
    logits=torch.full((1,4,64),-100.)
    logits.scatter_(2,initial[...,None],100.)
    sampled=(initial+1)%64
    with patch('llamafactory.train.diffusion.rl.denoise_step_logits',return_value=logits), \
         patch('llamafactory.train.diffusion.rl._commit',return_value=(sampled,logits,torch.zeros(1,4))):
        torch.manual_seed(42)
        actual,_=sample_canvas(model,torch.tensor([2,3]),steps=4,canvas_size=4,vocab_size=64)
    torch.testing.assert_close(actual,sampled)
