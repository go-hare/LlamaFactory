#!/usr/bin/env python3
"""Isolate AdamW implementations with identical precomputed gradient tensors."""

import argparse
import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


HYPERPARAMETERS = {"lr": 1.125e-4, "betas": [0.95, 0.99], "eps": 1e-8, "weight_decay": 1e-4}
TOLERANCE = {"atol": 1e-7, "rtol": 1e-5}


def frameworks():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch

    return {"jax": jax, "jnp": jnp, "np": np, "optax": optax, "torch": torch}


def metrics(fw, actual, reference):
    np = fw["np"]
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(a - b))),
        "relative_l2": float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30)),
        "within_fp32_tolerance": bool(np.allclose(a, b, **TOLERANCE)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
    }


def fixtures(fw, steps):
    np = fw["np"]
    rng = np.random.default_rng(20261118)
    initial = {
        "matrix": rng.normal(0, 0.03, (8, 16)).astype(np.float32),
        "norm_scale": np.linspace(0.75, 1.25, 16, dtype=np.float32),
        "layer_scale": np.ones(1, dtype=np.float32),
        "tiny": np.linspace(-1e-5, 1e-5, 16, dtype=np.float32),
    }
    gradients = []
    for step in range(steps):
        group = {}
        for index, (name, value) in enumerate(initial.items()):
            scale = (0.2, 0.02, 1e-7, 1e-9)[index]
            group[name] = rng.normal(0, scale, (4, *value.shape)).astype(np.float32)
            if step in (1, 2, 7):
                group[name].fill(0)
        gradients.append(group)
    return initial, gradients


def compare(fw, dtype, accumulation, steps=16, gradient_dtype=None):
    torch, jax, jnp, np, optax = (fw[key] for key in ("torch", "jax", "jnp", "np", "optax"))
    initial, gradients = fixtures(fw, steps)
    gradient_dtype = gradient_dtype or dtype
    parameters = {
        name: torch.nn.Parameter(torch.tensor(value, dtype=getattr(torch, gradient_dtype)).to(getattr(torch, dtype)))
        for name, value in initial.items()
    }
    jparams = {
        name: jnp.array(p.detach().float().numpy(), dtype=getattr(jnp, dtype), copy=True)
        for name, p in parameters.items()
    }
    if any(
        not np.array_equal(p.detach().float().numpy(), np.asarray(jparams[name], dtype=np.float32))
        for name, p in parameters.items()
    ):
        raise RuntimeError("Initial parameters differ")
    hp = HYPERPARAMETERS
    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=hp["lr"],
        betas=tuple(hp["betas"]),
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
        foreach=False,
    )
    transform = optax.adamw(
        hp["lr"], b1=hp["betas"][0], b2=hp["betas"][1], eps=hp["eps"], weight_decay=hp["weight_decay"]
    )
    state = transform.init(jparams)
    digest, records = hashlib.sha256(), []
    initial_values = {name: p.detach().float().numpy().copy() for name, p in parameters.items()}
    for index, group in enumerate(gradients, 1):
        optimizer.zero_grad(set_to_none=True)
        jgrads = {}
        for name, parameter in parameters.items():
            gradient = torch.zeros_like(parameter, dtype=getattr(torch, gradient_dtype))
            for micro in range(accumulation):
                gradient.add_(torch.tensor(group[name][micro], dtype=getattr(torch, gradient_dtype)) / accumulation)
            parameter.grad = gradient.to(parameter.dtype)
            values = parameter.grad.float().numpy().copy()
            digest.update(name.encode() + values.tobytes())
            jgrads[name] = jnp.array(values, dtype=getattr(jnp, dtype), copy=True)
            if not np.array_equal(values, np.asarray(jgrads[name], dtype=np.float32)):
                raise RuntimeError("Fixed gradients differ between frameworks")
        before_jax = jax.tree.map(lambda x: np.array(x, copy=True), jparams)
        before_torch = {name: p.detach().float().numpy().copy() for name, p in parameters.items()}
        optimizer.step()
        if any(not np.array_equal(before_jax[name], np.asarray(value)) for name, value in jparams.items()):
            raise RuntimeError("Torch mutated JAX parameter storage")
        updates, state = transform.update(jgrads, state, jparams)
        jparams = optax.apply_updates(jparams, updates)
        if int(state[0].count) != index:
            raise RuntimeError("Optax step count differs")
        details = []
        for name, parameter in parameters.items():
            ts = optimizer.state[parameter]
            if int(ts["step"]) != index:
                raise RuntimeError("Torch step count differs")
            item = {"parameter": name, "torch_step": int(ts["step"]), "optax_step": int(state[0].count)}
            for key, actual, reference in (
                ("weight", parameter.detach().float().numpy(), np.asarray(jparams[name], dtype=np.float32)),
                ("exp_avg", ts["exp_avg"].float().numpy(), np.asarray(state[0].mu[name], dtype=np.float32)),
                ("exp_avg_sq", ts["exp_avg_sq"].float().numpy(), np.asarray(state[0].nu[name], dtype=np.float32)),
                (
                    "update",
                    parameter.detach().float().numpy() - before_torch[name],
                    np.asarray(jparams[name], dtype=np.float32) - np.asarray(before_jax[name], dtype=np.float32),
                ),
            ):
                item[key] = metrics(fw, actual, reference)
            item["torch_moment_dtype"] = str(ts["exp_avg"].dtype)
            item["optax_moment_dtype"] = str(state[0].mu[name].dtype)
            item["torch_weight_changed"] = not np.array_equal(parameter.detach().float().numpy(), initial_values[name])
            item["optax_weight_changed"] = not np.array_equal(
                np.asarray(jparams[name], dtype=np.float32), initial_values[name]
            )
            details.append(item)
        records.append(
            {
                "step": index,
                "zero_gradient": all(not bool(torch.count_nonzero(p.grad)) for p in parameters.values()),
                "parameters": details,
            }
        )
    passed = all(
        item[key]["within_fp32_tolerance"] and item[key]["finite"]
        for record in records
        for item in record["parameters"]
        for key in ("weight", "exp_avg", "exp_avg_sq", "update")
    )
    return {
        "dtype": dtype,
        "gradient_dtype": gradient_dtype,
        "gradient_accumulation": accumulation,
        "steps": steps,
        "fixed_inputs_exactly_equal": True,
        "initial_parameters_sha256": hashlib.sha256(
            b"".join(initial_values[name].tobytes() for name in sorted(initial_values))
        ).hexdigest(),
        "shared_gradient_sha256": digest.hexdigest(),
        "torch_jax_storage_independence_checked": True,
        "within_fp32_tolerance": passed,
        "records": records,
        "scope": "Both optimizers receive the same accumulated, quantized gradient tensor; no model forward/backward or cross-framework gradient accumulation is involved",
    }


def run(steps=16):
    if steps < 3:
        raise ValueError("At least three steps are required to test nonzero then zero gradients")
    fw = frameworks()
    import optax
    import torch

    cases = [compare(fw, dtype, accumulation, steps) for dtype in ("float32", "bfloat16") for accumulation in (1, 4)]
    cases += [compare(fw, "float32", accumulation, steps, gradient_dtype="bfloat16") for accumulation in (1, 4)]
    return {
        "scope": "Controlled optimizer isolation for section-4 AR-to-diffusion conversion diagnostics; not Google's undisclosed conversion recipe or proof of generation quality",
        "hyperparameters": HYPERPARAMETERS,
        "fp32_tolerance": TOLERANCE,
        "acceptance": "Fixed tolerance set before comparison; outcomes are measured against it without relaxing it, including normal FP32 rounding that can exceed this strict update tolerance",
        "control": "Additional FP32 parameter/moment cases receive the identical BF16-quantized gradient stream; precision policy of actual training is unchanged",
        "versions": {name: version(name) for name in ("torch", "jax", "jaxlib", "optax", "numpy")},
        "implementation_sha256": {
            "torch_adamw": hashlib.sha256(inspect.getsource(torch.optim.AdamW).encode()).hexdigest(),
            "optax_adamw": hashlib.sha256(inspect.getsource(optax.adamw).encode()).hexdigest(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.steps)
    args.out.write_text(json.dumps(result, indent=2))
    for case in result["cases"]:
        worst = {
            key: max(item[key]["max_abs"] for record in case["records"] for item in record["parameters"])
            for key in ("weight", "exp_avg", "exp_avg_sq", "update")
        }
        print(
            "FIXED_GRADIENT_ADAMW",
            case["dtype"],
            "gradient_dtype",
            case["gradient_dtype"],
            "GA",
            case["gradient_accumulation"],
            "within_fp32_tolerance",
            case["within_fp32_tolerance"],
            worst,
            flush=True,
        )


if __name__ == "__main__":
    main()
