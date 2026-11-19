"""Independent PyTorch transcription of official preprocessing, not JAX parity."""
from collections import UserDict
import math
import pytest
import torch
import torch.nn.functional as F
from test_diffusion_ple import tiny_model
from llamafactory.train.diffusion.loss import (
    _diffusion_forward, _allow_mask, _gemma_masks, denoise_step_logits,
)


def rms(x, weight=None, eps=1e-6):
    y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)
    if weight is not None:y=y*weight.float()
    return y.to(x.dtype)


def reference(model, tokens, sc, start, masks, positions):
    text=model.model.language_model;cfg=text.config;cond=model.self_conditioner
    # Compute directly from tables and matrices; do not use diffusion hooks or
    # text.project_per_layer_inputs, nor SelfConditioner.forward.
    raw=F.embedding(tokens,text.embed_tokens.weight)*math.sqrt(cfg.hidden_size)
    projection=F.linear(raw,text.per_layer_model_projection.weight)*text.per_layer_model_projection_scale
    projection=projection.reshape(*tokens.shape,cfg.num_hidden_layers,cfg.hidden_size_per_layer_input)
    projection=rms(projection,text.per_layer_projection_norm.weight)
    identity=F.embedding(tokens[:,:start],text.embed_tokens_per_layer.weight)
    identity=identity*math.sqrt(cfg.hidden_size_per_layer_input)
    identity=identity.reshape(tokens.shape[0],start,cfg.num_hidden_layers,cfg.hidden_size_per_layer_input)
    ple=torch.cat([(projection[:,:start]+identity)*text.per_layer_input_scale,projection[:,start:]],1)
    normed=rms(sc,cond.pre_norm.weight)
    added=F.linear(F.gelu(F.linear(normed,cond.gate_proj.weight),approximate='tanh')*
                   F.linear(normed,cond.up_proj.weight),cond.down_proj.weight)
    canvas=rms(raw[:,start:]+added)
    hidden=torch.cat([raw[:,:start],canvas],1)
    rope={kind:text.rotary_emb(hidden,positions,kind) for kind in text.unique_layer_types}
    shared=UserDict()
    for i,layer in enumerate(text.layers):
        kind=cfg.layer_types[i]
        hidden=layer(hidden,ple[:,:,i],shared_kv_states=shared,position_embeddings=rope[kind],
                     attention_mask=masks[kind],position_ids=positions,past_key_values=None)
    logits=model.lm_head(text.norm(hidden))
    cap=cfg.final_logit_softcapping
    if cap is not None:logits=torch.tanh(logits/cap)*cap
    return logits


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('nonzero',[False,True])
def test_full_logits_match_formula(dtype,nonzero):
    model=tiny_model().to(dtype)
    model.config.image_token_id=10;model.config.audio_token_id=11;model.config.video_token_id=12
    tokens=torch.tensor([[2,3,4,10,11,12,7]])
    valid=torch.ones_like(tokens,dtype=torch.bool);start=3
    masks=_gemma_masks(_allow_mask(valid[:,:start],valid[:,start:],canvas_bidirectional=True),dtype,
                       valid=valid,sliding_window=16)
    positions=torch.arange(tokens.shape[1])[None]
    sc=torch.randn(1,4,16,dtype=dtype) if nonzero else torch.zeros(1,4,16,dtype=dtype)
    with torch.no_grad():
        actual=_diffusion_forward(model,tokens,sc,start,masks,positions).logits
        expected=reference(model,tokens,sc,start,masks,positions)
    max_diff=float((actual.float()-expected.float()).abs().max())
    print('REFERENCE',str(dtype),'SC',nonzero,'max_abs_diff',max_diff)
    torch.testing.assert_close(actual,expected,atol=2e-5 if dtype==torch.float32 else .004,
                               rtol=2e-5 if dtype==torch.float32 else .02)


def test_prompt_conditioning_and_bidirectionality():
    model=tiny_model();p=torch.tensor([2,3,4]);canvas=torch.tensor([5,6,7,8])
    zero=torch.zeros(1,4,16)
    with torch.no_grad():
        a=denoise_step_logits(model,p,canvas,sc_embeddings=zero,pad_id=0)
        b=denoise_step_logits(model,torch.tensor([2,20,21]),canvas,sc_embeddings=zero,pad_id=0)
        changed=canvas.clone();changed[-1]=22
        c=denoise_step_logits(model,p,changed,sc_embeddings=zero,pad_id=0)
    prompt_delta=float((a-b).abs().max());right_delta=float((a[:,0]-c[:,0]).abs().max())
    print('CONDITIONING prompt_delta',prompt_delta,'right_to_left_delta',right_delta)
    assert prompt_delta>1e-6
    assert right_delta>1e-6
    # Untie output head for this dependency test: otherwise its shared embedding
    # gradient would obscure which token embeddings input attention depends on.
    model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.zero_grad(set_to_none=True)
    logits=denoise_step_logits(model,p,canvas,sc_embeddings=zero,pad_id=0)
    F.cross_entropy(logits[:,0].float(),torch.tensor([9])).backward()
    grad=model.model.language_model.embed_tokens.weight.grad
    assert grad[p[1:]].abs().sum()>0
    print('CONDITIONING prompt_input_grad',float(grad[p[1:]].norm()))
