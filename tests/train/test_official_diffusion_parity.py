# ruff: noqa: E402

import os
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from compare_official_diffusion import differences, frameworks, run_case, to_torch_layout


def test_reference_source_changes_fail_before_import(tmp_path):
    (tmp_path / "changed.py").write_text("raise AssertionError('must not import')\n")
    with pytest.raises(RuntimeError, match="differs from pinned"):
        frameworks(tmp_path)


def test_layout_unpacking_preserves_distinct_gate_key_and_value_weights():
    fw = {"np": np}
    gate_up = np.stack([np.full((32, 16), 2), np.full((32, 16), 3)])
    np.testing.assert_array_equal(to_torch_layout(fw, "self_conditioner.gate_proj.weight", gate_up), gate_up[0])
    np.testing.assert_array_equal(to_torch_layout(fw, "self_conditioner.up_proj.weight", gate_up), gate_up[1])
    kv = np.arange(2 * 1 * 16 * 8).reshape(2, 1, 16, 8)
    np.testing.assert_array_equal(to_torch_layout(fw, "layers.0.self_attn.k_proj.weight", kv), kv[0, 0].T)
    np.testing.assert_array_equal(to_torch_layout(fw, "layers.0.self_attn.v_proj.weight", kv), kv[1, 0].T)
    stats = differences(fw, [1, 2], [1, 2])
    assert stats["max_abs"] == stats["relative_l2"] == 0


@pytest.mark.parametrize(
    "shared,backend,noise_pad",
    [(False, "eager", False), (True, "eager", False), (True, "sdpa", False), (False, "eager", True)],
)
def test_actual_official_fp32_execution_and_training_are_reported(shared, backend, noise_pad):
    source = os.environ.get("GEMMA_OFFICIAL_SRC")
    if not source:
        pytest.skip("Set GEMMA_OFFICIAL_SRC to the pinned official source; requires the separate JAX environment")
    pytest.importorskip("jax")
    fw = frameworks(Path(source))
    report = run_case(fw, shared, "float32", backend, noise_pad)
    assert report["kv_shared"] == shared
    assert report["torch_attention_backend"] == backend
    assert report["noise_contains_pad"] == noise_pad
    assert {case["sc"] for case in report["cases"]} == {"on", "off", "previous_pass"}
    for case in report["cases"]:
        assert case["logits"]["max_abs"] < 2e-5
        assert abs(case["torch_loss"] - case["official_loss"]) < 2e-5
        assert len(case["gradients"]) > 50
        assert all(item["within_fp32_tolerance"] for item in case["gradients"])
        assert max(case["official_only_trainable_scalar_gradients"].values()) > 1e-6
    assert report["torch_nontrainable_official_trainable"] == []
    assert any(item["parameter"].endswith(".layer_scalar") for item in report["cases"][0]["gradients"])
    assert report["full_training_equivalence"] is False
    assert [item["sc"] for item in report["updates"]] == ["on", "off", "off", "on"]
    for step in report["updates"]:
        assert step["grad_accum"] == 4
        for item in step["parameters_and_states"]:
            assert item["torch_step"] == item["official_step"] == step["step"]
            for key in ("parameter_difference", "exp_avg_difference", "exp_avg_sq_difference"):
                assert np.isfinite(item[key]["max_abs"])
    assert any(value != 1.0 for value in report["updates"][0]["official_layer_scales"].values())


def test_bf16_primitive_diagnostics_are_finite_and_do_not_patch_runtime():
    source = os.environ.get("GEMMA_OFFICIAL_SRC")
    if not source:
        pytest.skip("Requires the separate JAX environment and pinned official source")
    pytest.importorskip("jax")
    from diagnose_diffusion_bf16 import diagnose
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    original = Gemma4RMSNorm.forward
    report = diagnose(frameworks(Path(source)))
    assert Gemma4RMSNorm.forward is original
    assert report["production_math_changed"] is False
    assert report["probes"]["rms_norm"]["max_abs"] > 0
    assert all(item["actual_finite"] and item["official_finite"] for item in report["probes"].values())
