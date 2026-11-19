#!/usr/bin/env python3
"""Measure actual official versus HF BF16 primitives without changing runtime math."""
# ruff: noqa: E402

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from compare_official_diffusion import differences, frameworks


def diagnose(fw):
    jax, jnp, np, torch = (fw[key] for key in ("jax", "jnp", "np", "torch"))
    from gemma.gm.nn.gemma4 import _layers
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    from llamafactory.train.diffusion.loss import SelfConditioner

    values = np.random.default_rng(42).normal(0, 0.05, (2, 4, 16)).astype(np.float32)
    xj = jnp.asarray(values, dtype=jnp.bfloat16)
    xt = torch.tensor(values, dtype=torch.bfloat16)
    norm = _layers.RMSNorm()
    params = jax.tree.map(lambda p: p.astype(jnp.bfloat16), norm.init(jax.random.PRNGKey(42), xj))
    official_norm = norm.apply(params, xj)
    hf_norm = Gemma4RMSNorm(16).to(torch.bfloat16)
    actual_norm = hf_norm(xt)
    sc_norm = SelfConditioner(16, 32).to(torch.bfloat16).pre_norm(xt)
    official_variance = jnp.mean(jnp.square(xj), -1, keepdims=True)
    hf_variance = xt.float().square().mean(-1, keepdim=True)
    same_variance = torch.tensor(np.asarray(official_variance, dtype=np.float32), dtype=torch.bfloat16)
    denominator = same_variance + same_variance.new_tensor(1e-6)
    jax_denominator = official_variance + 1e-6
    probes = {
        "rms_norm": (actual_norm, official_norm),
        "sc_pre_norm": (sc_norm, official_norm),
        "variance_fp32_vs_bf16": (hf_variance, official_variance),
        "rsqrt_same_bf16_input": (torch.rsqrt(denominator), jax.lax.rsqrt(jax_denominator)),
        "gelu_tanh": (torch.nn.functional.gelu(xt, approximate="tanh"), jax.nn.gelu(xj)),
        "attention_softmax": (torch.softmax(xt.float(), -1).to(xt.dtype), jax.nn.softmax(xj, -1)),
    }
    reports = {}
    for name, (actual, expected) in probes.items():
        reports[name] = dict(
            differences(fw, actual.detach().float().numpy(), expected),
            actual_finite=bool(torch.isfinite(actual).all()),
            official_finite=bool(jnp.isfinite(expected).all()),
        )
    source = {
        "official_rms_norm": inspect.getsource(_layers.RMSNorm),
        "hf_rms_norm": inspect.getsource(Gemma4RMSNorm),
    }
    return {
        "scope": "BF16 primitive diagnostics on identical inputs; does not establish an official training precision policy or generation-quality cause",
        "production_math_changed": False,
        "probes": reports,
        "source": source,
        "source_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in source.items()},
        "decision": "Keep the stable HF numeric path; replacing a few operators with BF16 arithmetic has not established full-model/gradient equivalence",
    }


def diagnose_gradients(fw):
    jax, jnp, np, torch = (fw[key] for key in ("jax", "jnp", "np", "torch"))
    from gemma.gm.nn.gemma4 import _layers
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    atol, rtol = 2e-5, 2e-5
    shape = (2, 4, 16)
    base = np.random.default_rng(20261118).normal(size=shape).astype(np.float32)
    fixtures = {
        "ordinary": base * 0.05,
        "zero": np.zeros(shape, dtype=np.float32),
        "near_zero": base * 1e-5,
        "dynamic_range": base * np.geomspace(1e-4, 100, 16).astype(np.float32),
    }
    upstream = np.random.default_rng(781).normal(size=shape).astype(np.float32)
    scale = np.linspace(0.5, 1.5, 16, dtype=np.float32)

    def stats(actual, expected):
        actual = np.asarray(actual, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        return {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "relative_l2": float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-20)),
            "reference_max_abs": float(np.max(np.abs(expected))),
            "within_declared_fp32_tolerance": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "actual_finite": bool(np.isfinite(actual).all()),
            "reference_finite": bool(np.isfinite(expected).all()),
        }

    def staged_norm(x, weight, rsqrt_float):
        denominator = x.square().mean(-1, keepdim=True) + x.new_tensor(1e-6)
        reciprocal = torch.rsqrt(denominator.float()).to(x.dtype) if rsqrt_float else torch.rsqrt(denominator)
        return (x * reciprocal) * weight

    def staged_gelu(x):
        return (x * x.new_tensor(0.5)) * (
            x.new_tensor(1.0) + torch.tanh(x.new_tensor(np.sqrt(2 / np.pi)) * (x + x.new_tensor(0.044715) * (x**3)))
        )

    def staged_softmax(x):
        exponents = (x - x.amax(-1, keepdim=True)).exp()
        return exponents / exponents.sum(-1, keepdim=True)

    norm = _layers.RMSNorm()
    source = {
        "official_rmsnorm": inspect.getsource(_layers.RMSNorm),
        "official_gelu": inspect.getsource(jax.nn.gelu),
        "official_softmax": inspect.getsource(jax.nn.softmax),
        "hf_norm": inspect.getsource(Gemma4RMSNorm),
    }
    results = []
    for precision in ("float32", "bfloat16"):
        td, jd = getattr(torch, precision), getattr(jnp, precision)
        cotangent = torch.tensor(upstream, dtype=td)
        cj = jnp.array(cotangent.float().numpy(), dtype=jd, copy=True)
        wt = torch.tensor(scale, dtype=td)
        wj = jnp.array(wt.float().numpy(), dtype=jd, copy=True)
        for fixture, values in fixtures.items():
            quantized = torch.tensor(values, dtype=td)
            xj = jnp.array(quantized.float().numpy(), dtype=jd, copy=True)
            for operator in ("rms_norm", "softmax", "gelu"):
                if operator == "rms_norm":

                    def official(x, w):
                        return norm.apply({"params": {"scale": w}}, x)

                    variants = ("hf_fp32_norm", "torch_sc_rmsnorm", "staged_fp32_rsqrt", "staged_native_rsqrt")
                elif operator == "softmax":

                    def official(x, w):
                        return jax.nn.softmax(x, axis=-1)

                    variants = ("torch_fp32_softmax", "torch_native_softmax", "staged_softmax")
                else:

                    def official(x, w):
                        return jax.nn.gelu(x, approximate=True)

                    variants = ("torch_fused_gelu", "staged_gelu")

                def objective(x, w):
                    y = official(x, w)
                    return (y.astype(jnp.float32) * cj.astype(jnp.float32)).sum(), y

                (_, expected), (dx, dw) = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(xj, wj)
                for variant in variants:
                    x = quantized.detach().clone().requires_grad_(True)
                    weight = wt.detach().clone().requires_grad_(True)
                    if variant == "hf_fp32_norm":
                        module = Gemma4RMSNorm(16).to(td)
                        module.weight = torch.nn.Parameter(weight)
                        weight = module.weight
                        y = module(x)
                    elif variant == "torch_sc_rmsnorm":
                        module = torch.nn.RMSNorm(16, eps=1e-6).to(td)
                        module.weight = torch.nn.Parameter(weight)
                        weight = module.weight
                        y = module(x)
                    elif variant == "staged_fp32_rsqrt":
                        y = staged_norm(x, weight, True)
                    elif variant == "staged_native_rsqrt":
                        y = staged_norm(x, weight, False)
                    elif variant == "torch_fp32_softmax":
                        y = torch.softmax(x.float(), -1).to(td)
                    elif variant == "torch_native_softmax":
                        y = torch.softmax(x, -1)
                    elif variant == "staged_softmax":
                        y = staged_softmax(x)
                    elif variant == "torch_fused_gelu":
                        y = torch.nn.functional.gelu(x, approximate="tanh")
                    else:
                        y = staged_gelu(x)
                    total = (y.float() * cotangent.float()).sum()
                    gradients = torch.autograd.grad(total, (x, weight), allow_unused=True)
                    actual_dw = gradients[1] if gradients[1] is not None else torch.zeros_like(weight)
                    row = {
                        "dtype": precision,
                        "fixture": fixture,
                        "operator": operator,
                        "variant": variant,
                        "forward": stats(y.detach().float().numpy(), expected),
                        "input_gradient": stats(gradients[0].float().numpy(), dx),
                    }
                    if operator == "rms_norm":
                        row["scale_gradient"] = stats(actual_dw.float().numpy(), dw)
                    results.append(row)
                    print(
                        "PRIMITIVE",
                        precision,
                        fixture,
                        operator,
                        variant,
                        "forward",
                        row["forward"]["relative_l2"],
                        "dx",
                        row["input_gradient"]["relative_l2"],
                        "dw",
                        row.get("scale_gradient", {}).get("relative_l2"),
                        flush=True,
                    )
    summary = []
    baseline_variants = {"rms_norm": "hf_fp32_norm", "softmax": "torch_fp32_softmax", "gelu": "torch_fused_gelu"}
    for precision in ("float32", "bfloat16"):
        for operator in ("rms_norm", "softmax", "gelu"):
            group = [row for row in results if row["dtype"] == precision and row["operator"] == operator]
            baseline = baseline_variants[operator]
            for variant in sorted({row["variant"] for row in group}):
                rows = [row for row in group if row["variant"] == variant]
                metrics = (
                    ("forward", "input_gradient", "scale_gradient")
                    if operator == "rms_norm"
                    else ("forward", "input_gradient")
                )
                better = []
                for row in rows:
                    before = next(
                        item for item in group if item["variant"] == baseline and item["fixture"] == row["fixture"]
                    )
                    better.append(
                        all(row[key]["relative_l2"] <= before[key]["relative_l2"] + 1e-12 for key in metrics)
                    )
                summary.append(
                    {
                        "dtype": precision,
                        "operator": operator,
                        "variant": variant,
                        "all_fixtures_no_worse_forward_and_gradients": all(better),
                        "fixtures_no_worse": sum(better),
                        "fixtures": len(rows),
                        "all_within_declared_fp32_tolerance": all(
                            row[key]["within_declared_fp32_tolerance"] for row in rows for key in metrics
                        ),
                        "max_relative_l2": {key: max(row[key]["relative_l2"] for row in rows) for key in metrics},
                    }
                )
    return {
        "scope": "Identical quantized input, nonunit scale, quantized upstream cotangent; no optimizer/model fitting",
        "production_math_changed": False,
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
            "policy": "Declared before first run; the same strict FP32 criterion is reported for BF16 without widening",
        },
        "fixtures": {key: value.tolist() for key, value in fixtures.items()},
        "scale": scale.tolist(),
        "upstream": upstream.tolist(),
        "records": results,
        "summary": summary,
        "all_outputs_and_gradients_finite": all(
            row[key]["actual_finite"] and row[key]["reference_finite"]
            for row in results
            for key in ("forward", "input_gradient", "scale_gradient")
            if key in row
        ),
        "source": source,
        "source_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in source.items()},
        "full_model_promotion": "Isolated equality does not establish full-model or gradient equivalence",
    }


def diagnose_full_model(fw):
    from types import MethodType
    from unittest.mock import patch

    import compare_official_diffusion as comparison
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    torch = fw["torch"]
    original_factory = comparison.make_models
    reports = []

    def staged_norm(module, value):
        variance = value.square().mean(-1, keepdim=True) + value.new_tensor(module.eps)
        output = value * torch.rsqrt(variance.float()).to(value.dtype)
        weight = getattr(module, "weight", None)
        return output if weight is None else output * weight

    for variant in ("baseline", "staged_norm"):

        def make(*args, **kwargs):
            model, official = original_factory(*args, **kwargs)
            if variant == "staged_norm":
                for module in model.modules():
                    if isinstance(module, Gemma4RMSNorm | torch.nn.RMSNorm):
                        module.forward = MethodType(staged_norm, module)
            return model, official

        with patch.object(comparison, "make_models", side_effect=make):
            report = comparison.run_case(fw, False, "bfloat16", attention_backend="eager")
        reports.append({"variant": variant, "report": report})
    baseline, candidate = [entry["report"] for entry in reports]
    no_worse = []
    for before, after in zip(baseline["cases"], candidate["cases"]):
        gradients_before = {item["parameter"]: item for item in before["gradients"]}
        no_worse.append(
            after["logits"]["relative_l2"] <= before["logits"]["relative_l2"]
            and after["logits"]["max_abs"] <= before["logits"]["max_abs"]
            and all(
                item["max_abs"] <= gradients_before[item["parameter"]]["max_abs"]
                and item["relative_l2"] <= gradients_before[item["parameter"]]["relative_l2"]
                for item in after["gradients"]
            )
        )
    assert comparison.make_models is original_factory
    return {
        "scope": "Actual official tiny JAX versus HF; candidate methods are bound only to temporary model instances",
        "production_math_changed": False,
        "factory_restored": True,
        "reports": reports,
        "all_cases_no_worse_outputs_and_gradients": all(no_worse),
        "production_promotion": False,
        "decision": "No automatic promotion: all output and gradient cases must improve and remain stable; this is CPU diagnostic evidence only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gradients", action="store_true", help="Compare 72 primitive forward/backward cases")
    mode.add_argument(
        "--full-model-candidate", action="store_true", help="Test staged norm on temporary model instances"
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    fw = frameworks(args.official_src)
    result = (
        diagnose_full_model(fw)
        if args.full_model_candidate
        else diagnose_gradients(fw)
        if args.gradients
        else diagnose(fw)
    )
    result["diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[1]
    result["local_source_sha256"] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("scripts/compare_official_diffusion.py", "src/llamafactory/train/diffusion/loss.py")
    }
    args.out.write_text(json.dumps(result, indent=2))
    if args.full_model_candidate:
        print(
            "FULL_MODEL_CANDIDATE",
            result["all_cases_no_worse_outputs_and_gradients"],
            "production_promotion",
            result["production_promotion"],
            flush=True,
        )
    else:
        print(json.dumps(result["summary"] if args.gradients else result["probes"], indent=2), flush=True)


if __name__ == "__main__":
    main()
