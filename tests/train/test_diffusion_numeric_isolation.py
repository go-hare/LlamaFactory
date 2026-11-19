# ruff: noqa: E402

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from isolate_diffusion_optimizer import TOLERANCE, compare, fixtures, frameworks, run


def test_optimizer_budget_requires_nonzero_then_zero_updates():
    with pytest.raises(ValueError, match="three steps"):
        run(2)


def test_fixed_gradient_isolation_inputs_counts_and_precision_controls():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    fw = frameworks()
    report = run(steps=8)
    assert report["fp32_tolerance"] == TOLERANCE == {"atol": 1e-7, "rtol": 1e-5}
    assert len(report["cases"]) == 6
    for case in report["cases"]:
        assert case["fixed_inputs_exactly_equal"] and case["torch_jax_storage_independence_checked"]
        assert [r["step"] for r in case["records"]] == list(range(1, 9))
        assert [r["step"] for r in case["records"] if r["zero_gradient"]] == [2, 3, 8]
        for record in case["records"]:
            assert len(record["parameters"]) == 4
            for item in record["parameters"]:
                assert item["torch_step"] == item["optax_step"] == record["step"]
                assert all(item[key]["finite"] for key in ("weight", "exp_avg", "exp_avg_sq", "update"))
        if case["dtype"] == "float32":
            assert max(item["weight"]["max_abs"] for r in case["records"] for item in r["parameters"]) < 1e-6
            assert max(item["exp_avg"]["max_abs"] for r in case["records"] for item in r["parameters"]) < 5e-8
    for accumulation in (1, 4):
        native = next(
            c for c in report["cases"] if c["dtype"] == "bfloat16" and c["gradient_accumulation"] == accumulation
        )
        fp32 = next(
            c
            for c in report["cases"]
            if c["dtype"] == "float32"
            and c["gradient_dtype"] == "bfloat16"
            and c["gradient_accumulation"] == accumulation
        )
        assert native["initial_parameters_sha256"] == fp32["initial_parameters_sha256"]
        assert native["shared_gradient_sha256"] == fp32["shared_gradient_sha256"]
        native_error = max(item["exp_avg"]["max_abs"] for r in native["records"] for item in r["parameters"])
        fp32_error = max(item["exp_avg"]["max_abs"] for r in fp32["records"] for item in r["parameters"])
        assert fp32_error < native_error
    initial, gradients = fixtures(fw, 8)
    assert set(initial) == {"matrix", "norm_scale", "layer_scale", "tiny"}
    assert len(gradients) == 8
    repeated = compare(fw, "float32", 4, steps=8)
    expected = next(
        c
        for c in report["cases"]
        if c["dtype"] == "float32" and c["gradient_dtype"] == "float32" and c["gradient_accumulation"] == 4
    )
    assert repeated == expected


def test_primitive_candidates_report_forward_and_backward_without_promotion():
    source = os.environ.get("GEMMA_OFFICIAL_SRC")
    if not source:
        pytest.skip("Requires the separate JAX environment and pinned official source")
    pytest.importorskip("jax")
    from compare_official_diffusion import frameworks as official_frameworks
    from diagnose_diffusion_bf16 import diagnose_gradients
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    original = Gemma4RMSNorm.forward
    report = diagnose_gradients(official_frameworks(Path(source)))
    assert Gemma4RMSNorm.forward is original
    assert report["production_math_changed"] is False
    assert report["all_outputs_and_gradients_finite"]
    assert len(report["records"]) == 72 and len(report["summary"]) == 18
    assert report["tolerances"]["atol"] == report["tolerances"]["rtol"] == 2e-5
    assert set(report["fixtures"]) == {"ordinary", "zero", "near_zero", "dynamic_range"}
    assert all(
        summary["all_within_declared_fp32_tolerance"] for summary in report["summary"] if summary["dtype"] == "float32"
    )
    staged = [
        row
        for row in report["records"]
        if row["dtype"] == "bfloat16" and row["variant"] in ("staged_fp32_rsqrt", "staged_softmax", "staged_gelu")
    ]
    assert all(row["forward"]["max_abs"] == 0 for row in staged)
    assert any(not row["input_gradient"]["within_declared_fp32_tolerance"] for row in staged)
    gelu = next(
        summary
        for summary in report["summary"]
        if summary["dtype"] == "bfloat16" and summary["variant"] == "staged_gelu"
    )
    softmax = next(
        summary
        for summary in report["summary"]
        if summary["dtype"] == "bfloat16" and summary["variant"] == "staged_softmax"
    )
    assert not gelu["all_fixtures_no_worse_forward_and_gradients"]
    assert not softmax["all_fixtures_no_worse_forward_and_gradients"]
