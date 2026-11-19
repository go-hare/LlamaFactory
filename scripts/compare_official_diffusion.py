#!/usr/bin/env python3
"""Compare actual official Flax execution with the local HF diffusion forward."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REVISION = "9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3"
OFFICIAL_PYTHON_SHA256 = "9fd50695d3f593d374183bde49fdef8f95e68ccc41948c0b4b97824a3524e9fe"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
sys.path.insert(0, str(ROOT / "src"))


def frameworks(official):
    fingerprint = hashlib.sha256()
    for path in sorted(official.rglob("*.py")):
        fingerprint.update(str(path.relative_to(official)).encode() + b"\0" + path.read_bytes())
    if fingerprint.hexdigest() != OFFICIAL_PYTHON_SHA256:
        raise RuntimeError("Official Python source differs from pinned revision; no comparison started")
    sys.path.insert(0, str(official))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch
    from flax.traverse_util import flatten_dict, unflatten_dict
    from gemma.diffusion import _models, _sampler
    from gemma.gm.nn.gemma4 import _config, _modules
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    from llamafactory.train.diffusion.loss import (
        _allow_mask,
        _diffusion_forward,
        _embed_distribution,
        _gemma_masks,
        _get_conditioner,
        _token_embedding,
        prepare_conversion_training,
    )

    for module in (_models, _sampler, _config, _modules):
        if not Path(module.__file__).resolve().is_relative_to(official.resolve()):
            raise RuntimeError(f"Official module resolved outside requested source: {module.__file__}")
    return dict(
        jax=jax,
        jnp=jnp,
        np=np,
        optax=optax,
        torch=torch,
        flatten_dict=flatten_dict,
        unflatten_dict=unflatten_dict,
        _models=_models,
        _sampler=_sampler,
        _config=_config,
        _modules=_modules,
        Gemma4Config=Gemma4Config,
        Gemma4ForConditionalGeneration=Gemma4ForConditionalGeneration,
        Gemma4TextConfig=Gemma4TextConfig,
        _diffusion_forward=_diffusion_forward,
        _get_conditioner=_get_conditioner,
        _allow_mask=_allow_mask,
        _gemma_masks=_gemma_masks,
        _embed_distribution=_embed_distribution,
        _token_embedding=_token_embedding,
        prepare_conversion_training=prepare_conversion_training,
    )


def make_models(fw, shared, dtype, attention_backend="eager"):
    torch, jnp = fw["torch"], fw["jnp"]
    types = ["sliding_attention", "full_attention"] * 2
    cfg = fw["Gemma4TextConfig"](
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        hidden_size_per_layer_input=4,
        sliding_window=8,
        layer_types=types,
        num_kv_shared_layers=2 if shared else 0,
        use_double_wide_mlp=shared,
        final_logit_softcapping=30.0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(1234)
    hf = fw["Gemma4ForConditionalGeneration"](fw["Gemma4Config"](text_config=cfg)).to(getattr(torch, dtype))
    hf.set_attn_implementation(attention_backend)
    if hf.model.language_model.config._attn_implementation != attention_backend:
        raise RuntimeError("HF did not select the requested attention backend")
    fw["prepare_conversion_training"](hf)
    modules, config = fw["_modules"], fw["_config"]
    sharing = (
        config.KVCacheSharingConfig(frac_shared_layers=0.5, share_global=True, share_local=True) if shared else None
    )
    jc = config.TransformerConfig(
        num_embed=64,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        global_key_size=8,
        attention_types=(modules.AttentionType.LOCAL_SLIDING, modules.AttentionType.GLOBAL) * 2,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcap=30.0,
        sliding_window_size=8,
        per_layer_input_dim=4,
        global_rope_proportion=0.25,
        local_rope_proportion=1.0,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        kv_cache_sharing_config=sharing,
        override_kv_shared_ffw_hidden=64 if shared else None,
    )
    model = fw["_models"].DiffusionGemma_26B_A4B(config=jc, dtype=getattr(jnp, dtype), text_only=True)
    return hf, model


def torch_to_jax(fw, hf, template):
    np, jnp = fw["np"], fw["jnp"]
    state = hf.state_dict()
    flat = fw["flatten_dict"](template)
    mapped, links = {}, {}

    def assign(path, source, transform=lambda x: x):
        path = tuple(path.split("/"))
        value = transform(state[source].detach().float().numpy())
        if path not in flat or tuple(value.shape) != tuple(flat[path].shape):
            raise ValueError(f"Mapping shape mismatch: {path}, {value.shape}")
        mapped[path] = jnp.array(value, dtype=flat[path].dtype, copy=True)
        links[source] = path

    prefix = "model.language_model."
    assign("embedder/input_embedding", prefix + "embed_tokens.weight")
    assign("embedder/per_layer_embeddings", prefix + "embed_tokens_per_layer.weight", lambda x: x.reshape(64, 4, 4))
    assign(
        "embedder/per_layer_model_projection/w",
        prefix + "per_layer_model_projection.weight",
        lambda x: x.T.reshape(16, 4, 4),
    )
    assign("embedder/per_layer_projection_norm/scale", prefix + "per_layer_projection_norm.weight")
    assign("final_norm/scale", prefix + "norm.weight")
    for i, layer in enumerate(hf.model.language_model.layers):
        p, q = f"{prefix}layers.{i}.", f"layer_{i}/"
        assign(q + "skip_scale", p + "layer_scalar")
        assign(q + "attn/q_einsum/w", p + "self_attn.q_proj.weight", lambda x: x.reshape(2, 8, 16).transpose(0, 2, 1))
        assign(q + "attn/query_norm/scale", p + "self_attn.q_norm.weight")
        assign(q + "attn/attn_vec_einsum/w", p + "self_attn.o_proj.weight", lambda x: x.T.reshape(2, 8, 16))
        if not layer.self_attn.is_kv_shared_layer:
            key = tuple((q + "attn/kv_einsum/w").split("/"))
            values = np.stack(
                [
                    state[p + "self_attn." + name + ".weight"]
                    .detach()
                    .float()
                    .numpy()
                    .reshape(1, 8, 16)
                    .transpose(0, 2, 1)
                    for name in ("k_proj", "v_proj")
                ]
            )
            mapped[key] = jnp.asarray(values, dtype=flat[key].dtype)
            links[p + "self_attn.k_proj.weight"] = key
            links[p + "self_attn.v_proj.weight"] = key
            assign(q + "attn/key_norm/scale", p + "self_attn.k_norm.weight")
        gate_key = tuple((q + "mlp/gating_einsum").split("/"))
        gates = np.stack(
            [state[p + "mlp." + name + ".weight"].detach().float().numpy() for name in ("gate_proj", "up_proj")]
        )
        mapped[gate_key] = jnp.asarray(gates, dtype=flat[gate_key].dtype)
        links[p + "mlp.gate_proj.weight"] = gate_key
        links[p + "mlp.up_proj.weight"] = gate_key
        assign(q + "mlp/linear", p + "mlp.down_proj.weight", lambda x: x.T)
        for source, target in (
            ("input_layernorm", "pre_attention_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("pre_feedforward_layernorm", "pre_ffw_norm"),
            ("post_feedforward_layernorm", "post_ffw_norm"),
            ("post_per_layer_input_norm", "post_per_layer_input_norm"),
        ):
            assign(q + target + "/scale", p + source + ".weight")
        assign(q + "per_layer_input_gate/w", p + "per_layer_input_gate.weight", lambda x: x.T)
        assign(q + "per_layer_projection/w", p + "per_layer_projection.weight", lambda x: x.T)
    assign("self_conditioner/pre_norm/scale", "self_conditioner.pre_norm.weight")
    key = ("self_conditioner", "ffw", "gating_einsum")
    mapped[key] = jnp.asarray(
        np.stack(
            [
                state["self_conditioner." + name + ".weight"].detach().float().numpy()
                for name in ("gate_proj", "up_proj")
            ]
        ),
        dtype=flat[key].dtype,
    )
    links["self_conditioner.gate_proj.weight"] = key
    links["self_conditioner.up_proj.weight"] = key
    assign("self_conditioner/ffw/linear", "self_conditioner.down_proj.weight", lambda x: x.T)
    missing = {name for name, _ in hf.named_parameters()} - set(links)
    if missing:
        raise ValueError(f"Unmapped HF trainable parameters: {missing}")
    extras = set(flat) - set(mapped)
    if any(not ("attn" in key and key[2] in ("kv_einsum", "key_norm")) for key in extras):
        raise ValueError(f"Unexpected official-only parameters: {extras}")
    for key in extras:
        mapped[key] = flat[key]
    return fw["unflatten_dict"](mapped), links, extras


def to_torch_layout(fw, source, value):
    np = fw["np"]
    x = np.asarray(value, dtype=np.float32)
    if source.endswith("embed_tokens_per_layer.weight"):
        return x.reshape(64, 16)
    if source.endswith("per_layer_model_projection.weight"):
        return x.reshape(16, 16).T
    if ".self_attn.q_proj.weight" in source:
        return x.transpose(0, 2, 1).reshape(16, 16)
    if ".self_attn.k_proj.weight" in source or ".self_attn.v_proj.weight" in source:
        idx = 0 if source.endswith("k_proj.weight") else 1
        return x[idx].transpose(0, 2, 1).reshape(8, 16)
    if source.endswith("self_attn.o_proj.weight"):
        return x.reshape(16, 16).T
    if source.endswith("gate_proj.weight") or source.endswith("up_proj.weight"):
        return x[0 if source.endswith("gate_proj.weight") else 1]
    if source.endswith(("down_proj.weight", "per_layer_input_gate.weight", "per_layer_projection.weight")):
        return x.T
    return x


def differences(fw, actual, expected):
    np = fw["np"]
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "relative_l2": float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12)),
        "reference_max_abs": float(np.max(np.abs(b))),
        "within_fp32_tolerance": bool(np.allclose(a, b, atol=2e-4, rtol=1e-4)),
    }


def run_case(fw, shared, dtype, attention_backend="eager", noise_pad=False):
    torch, jax, jnp, np = (fw[k] for k in ("torch", "jax", "jnp", "np"))
    hf, model = make_models(fw, shared, dtype, attention_backend)
    history = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=np.int32)
    canvas = np.array([[0 if noise_pad else 21, 22, 23, 24]], dtype=np.int32)
    target = np.array([[31, 32, 33, 1]], dtype=np.int32)
    jd = getattr(jnp, dtype)
    zero = jnp.zeros((1, 4, 16), dtype=jd)
    cache = model.config.init_cache(batch_size=1, dtype=jd, cache_length=16)
    positions = jnp.arange(12)[None]
    causal = jnp.pad(jnp.tril(jnp.ones((1, 12, 12), dtype=jnp.bool_)), ((0, 0), (0, 0), (0, 4)))
    mask = fw["_sampler"]._make_global_attention_mask(1, 4, 16, jnp.array([12]))
    sliding = fw["_sampler"]._make_block_local_attention_mask(1, 4, 8, 16, jnp.array([12]))

    def initialize(module):
        module(tokens=jnp.asarray(history), cache=cache, positions=positions, attention_mask=causal)
        return module.call_with_self_conditioning(
            tokens=jnp.asarray(canvas),
            sc_embeddings=zero,
            cache=cache,
            positions=jnp.arange(12, 16)[None],
            attention_mask=mask,
            sliding_attention_mask=sliding,
        )

    template = model.init(jax.random.PRNGKey(42), method=initialize)["params"]
    params, links, extras = torch_to_jax(fw, hf, template)
    print("PARAMETERS", shared, dtype, len(links), "official_only", ["/".join(p) for p in sorted(extras)], flush=True)

    def forward(parameters, signal):
        encoded = model.apply(
            {"params": parameters},
            tokens=jnp.asarray(history),
            cache=cache,
            positions=positions,
            attention_mask=causal,
        )
        return model.apply(
            {"params": parameters},
            tokens=jnp.asarray(canvas),
            sc_embeddings=signal,
            cache=encoded.cache,
            positions=jnp.arange(12, 16)[None],
            attention_mask=mask,
            sliding_attention_mask=sliding,
            method=model.call_with_self_conditioning,
        ).logits

    tokens = torch.tensor(np.concatenate([history, canvas], axis=1), dtype=torch.long)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    masks = fw["_gemma_masks"](
        fw["_allow_mask"](valid[:, :12], valid[:, 12:], canvas_bidirectional=True),
        getattr(torch, dtype),
        valid=valid,
        sliding_window=8,
    )
    torch_positions = torch.arange(16)[None]

    def local_logits(signal, derived=False):
        signal = torch.tensor(np.asarray(signal, dtype=np.float32), dtype=getattr(torch, dtype))
        if derived:
            with torch.no_grad():
                first = local_logits(zero)
            signal = fw["_embed_distribution"](fw["_token_embedding"](hf), first)
        return fw["_diffusion_forward"](hf, tokens, signal, 12, masks, torch_positions).logits[:, 12:]

    def objective(parameters, signal, derived=False):
        if derived:
            first = jax.lax.stop_gradient(forward(parameters, zero))
            signal = model.apply({"params": parameters}, first, method=lambda m, x: m.embedder.encode_logits(x))
        logits = forward(parameters, signal)
        return fw["optax"].softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(target)).mean(), logits

    value_grad = jax.jit(jax.value_and_grad(objective, has_aux=True), static_argnums=2)
    records = []
    for name, signal in (
        ("off", zero),
        ("on", jnp.asarray(np.random.default_rng(42).normal(size=(1, 4, 16)), dtype=jd)),
        ("previous_pass", zero),
    ):
        hf.zero_grad(set_to_none=True)
        derived = name == "previous_pass"
        actual = local_logits(signal, derived)
        loss = torch.nn.functional.cross_entropy(
            actual.reshape(-1, 64), torch.tensor(target.reshape(-1), dtype=torch.long)
        )
        loss.backward()

        (jl, predicted), grads = value_grad(params, signal, derived)
        gradient = fw["flatten_dict"](grads)
        details = []
        for source, parameter in hf.named_parameters():
            expected = to_torch_layout(fw, source, gradient[links[source]])
            if parameter.grad is None:
                raise RuntimeError(f"Missing HF gradient: {source}")
            details.append(
                dict(parameter=source, **differences(fw, parameter.grad.detach().float().numpy(), expected))
            )
        scalar_grads = {
            f"layer_{i}/skip_scale": float(jnp.max(jnp.abs(grads[f"layer_{i}"]["skip_scale"]))) for i in range(4)
        }
        record = {
            "sc": name,
            "logits": differences(fw, actual.detach().float().numpy(), predicted),
            "torch_loss": float(loss.detach()),
            "official_loss": float(jl),
            "gradients": details,
            "official_only_trainable_scalar_gradients": scalar_grads,
        }
        records.append(record)
        print(
            "PARITY",
            shared,
            dtype,
            name,
            "logits",
            record["logits"],
            "gradient_max",
            max(d["max_abs"] for d in details),
            flush=True,
        )
    optimizer = torch.optim.AdamW(hf.parameters(), lr=1.125e-4, betas=(0.95, 0.99), weight_decay=1e-4, foreach=False)
    transformation = fw["optax"].adamw(1.125e-4, b1=0.95, b2=0.99, weight_decay=1e-4)
    optimizer_state = transformation.init(params)
    update_records = []
    on = jnp.asarray(np.random.default_rng(42).normal(size=(1, 4, 16)), dtype=jd)
    for step, enabled in enumerate((True, False, False, True), 1):
        signal = on if enabled else zero
        hf.zero_grad(set_to_none=True)
        accumulated = jax.tree.map(jnp.zeros_like, params)
        for _ in range(4):
            logits = local_logits(signal, enabled)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 64), torch.tensor(target.reshape(-1), dtype=torch.long)
            )
            (loss / 4).backward()
            (_, _), micro_grad = value_grad(params, signal, enabled)
            accumulated = jax.tree.map(lambda total, g: total + g / 4, accumulated, micro_grad)
        before_torch_step = jax.tree.map(lambda x: np.array(x, copy=True), params)
        optimizer.step()
        for before, after in zip(jax.tree.leaves(before_torch_step), jax.tree.leaves(params)):
            if not np.array_equal(before, np.asarray(after)):
                raise RuntimeError("PyTorch update mutated JAX weights through shared storage")
        updates, optimizer_state = transformation.update(accumulated, optimizer_state, params)
        params = fw["optax"].apply_updates(params, updates)
        flat_params = fw["flatten_dict"](params)
        mu, nu = fw["flatten_dict"](optimizer_state[0].mu), fw["flatten_dict"](optimizer_state[0].nu)
        details = []
        for name, parameter in hf.named_parameters():
            state = optimizer.state[parameter]
            item = {
                "parameter": name,
                "torch_step": int(state["step"]),
                "official_step": int(optimizer_state[0].count),
            }
            for key, actual, expected in (
                ("parameter", parameter.detach().float().numpy(), flat_params[links[name]]),
                ("exp_avg", state["exp_avg"].float().numpy(), mu[links[name]]),
                ("exp_avg_sq", state["exp_avg_sq"].float().numpy(), nu[links[name]]),
            ):
                item[key + "_difference"] = differences(fw, actual, to_torch_layout(fw, name, expected))
            details.append(item)
        scalar_updates = {f"layer_{i}/skip_scale": float(params[f"layer_{i}"]["skip_scale"][0]) for i in range(4)}
        update_records.append(
            {
                "step": step,
                "sc": "on" if enabled else "off",
                "grad_accum": 4,
                "parameters_and_states": details,
                "official_layer_scales": scalar_updates,
            }
        )
        print(
            "UPDATE",
            shared,
            dtype,
            step,
            enabled,
            "max_parameter_difference",
            max(d["parameter_difference"]["max_abs"] for d in details),
            "official_layer_scales",
            scalar_updates,
            flush=True,
        )
    return {
        "kv_shared": shared,
        "dtype": dtype,
        "cases": records,
        "updates": update_records,
        "full_training_equivalence": False,
        "equivalence_blocker": "Full E2B/CUDA training and undisclosed section-4 recipe remain outside this controlled tiny-model comparison",
        "optimizer_scope": "Same experimental AdamW on each framework's natural parameter tree, constant LR, no warmup or gradient clipping; not a published section-4 recipe",
        "optimizer_hyperparameters": {"lr": 1.125e-4, "betas": [0.95, 0.99], "eps": 1e-8, "weight_decay": 1e-4},
        "inputs": {"history": history.tolist(), "canvas": canvas.tolist(), "target": target.tolist(), "sc_seed": 42},
        "torch_attention_backend": hf.model.language_model.config._attn_implementation,
        "noise_contains_pad": noise_pad,
        "jax_devices": [str(device) for device in jax.devices()],
        "parameter_storage_independence_checked_each_step": True,
        "fp32_tolerance": {
            "atol": 2e-4,
            "rtol": 1e-4,
            "scope": "recorded separately for every compared array; not a CUDA or BF16 parity claim",
        },
        "official_only_unused_kv_parameters": ["/".join(p) for p in sorted(extras)],
        "torch_nontrainable_official_trainable": [
            source for source in links if source not in dict(hf.named_parameters())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--noise-pad", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    fw = frameworks(args.official_src)
    result = run_case(fw, args.shared, args.dtype, args.attention_backend, args.noise_pad)
    result["versions"] = {
        p: version(p) for p in ("jax", "jaxlib", "flax", "kauldron", "optax", "torch", "transformers")
    }
    result["official_revision"] = OFFICIAL_REVISION
    result["local_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("scripts/compare_official_diffusion.py", "src/llamafactory/train/diffusion/loss.py")
    }
    result["official_file_hashes"] = {
        str(p.relative_to(args.official_src)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.official_src.rglob("*.py")
        if "__pycache__" not in str(p)
    }
    result["scope"] = (
        "Actual official Flax cache prefill+diffusion forward, all HF trainable gradients, and four GA=4 AdamW updates; official-only trainable scalars reported explicitly"
    )
    import kauldron

    compatibility_file = Path(kauldron.__file__).parent / "ktyping/dtypes.py"
    compatibility_text = compatibility_file.read_text()
    compatibility_line = 'float128 = NpDType(np.float128) if hasattr(np, "float128") else NpDType(np.longdouble)'
    result["dependency_compatibility"] = {
        "platform": sys.platform,
        "tensorflow_distribution": version("tensorflow"),
        "kauldron_dtype_compatibility_applied": compatibility_line in compatibility_text,
        "kauldron_dtype_file_sha256": hashlib.sha256(compatibility_file.read_bytes()).hexdigest(),
        "patch": compatibility_line if compatibility_line in compatibility_text else None,
        "scope": "Mac type-checking alias only; official Gemma model source hash verified unchanged",
    }
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
