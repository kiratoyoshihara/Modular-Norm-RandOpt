from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.perturbation_norms import (  # noqa: E402
    load_mass_config,
    natural_norm,
    normalization_denominator,
)
from utils.recursive_modular_v1 import (  # noqa: E402
    build_recursive_modular_result,
    build_recursive_modular_result_rmsnorm_only,
)
from utils.worker_extn_ablation import WorkerExtension  # noqa: E402


def _parameter(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape, generator=generator))


def _named_parameters(num_layers=3, include_head=True):
    rows = [("model.embed_tokens.weight", _parameter((32, 8), 1))]
    seed = 2
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        entries = [
            (f"{prefix}.input_layernorm.weight", (8,)),
            (f"{prefix}.self_attn.q_proj.weight", (8, 8)),
            (f"{prefix}.self_attn.k_proj.weight", (4, 8)),
            (f"{prefix}.self_attn.v_proj.weight", (4, 8)),
            (f"{prefix}.self_attn.o_proj.weight", (8, 8)),
            (f"{prefix}.post_attention_layernorm.weight", (8,)),
            (f"{prefix}.mlp.gate_proj.weight", (16, 8)),
            (f"{prefix}.mlp.up_proj.weight", (16, 8)),
            (f"{prefix}.mlp.down_proj.weight", (8, 16)),
        ]
        for name, shape in entries:
            rows.append((name, _parameter(shape, seed)))
            seed += 1
    rows.append(("model.norm.weight", _parameter((8,), seed)))
    if include_head:
        rows.append(("lm_head.weight", _parameter((32, 8), seed + 1)))
    return rows


def test_exactly_two_rmsnorm_parameters_per_layer_are_touched():
    named = _named_parameters(num_layers=3)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(named, config, rmsnorm_multiplier=2.0)

    changed = {name for name in v1.scales if v1.scales[name] != rms.scales[name]}
    expected = {
        name
        for name in v1.scales
        if name.endswith("input_layernorm.weight") or name.endswith("post_attention_layernorm.weight")
    }
    assert changed == expected
    assert len(changed) == 2 * 3
    assert rms.diagnostics["num_rmsnorm_parameters_scaled"] == 2 * 3


def test_non_target_parameters_are_bit_identical_to_v1():
    named = _named_parameters(num_layers=3)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(named, config, rmsnorm_multiplier=2.0)

    target_suffixes = ("input_layernorm.weight", "post_attention_layernorm.weight")
    for name in v1.scales:
        if name.endswith(target_suffixes):
            continue
        assert v1.scales[name] == rms.scales[name], name


def test_final_norm_and_head_are_untouched():
    named = _named_parameters(num_layers=3)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(named, config, rmsnorm_multiplier=2.0)

    for name in ("model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"):
        assert math.isclose(rms.scales[name], v1.scales[name], rel_tol=1e-12)


@pytest.mark.parametrize("multiplier", [2.0, 3.0, 0.5])
def test_target_scale_ratio_equals_multiplier_exactly(multiplier):
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(
        named, config, rmsnorm_multiplier=multiplier
    )
    for name in v1.scales:
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            assert math.isclose(rms.scales[name] / v1.scales[name], multiplier, rel_tol=1e-12)


def test_diagnostics_do_not_claim_sensitivity_profile_usage():
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    rms = build_recursive_modular_result_rmsnorm_only(named, config)
    assert rms.diagnostics["sensitivity_profile_used"] is False
    assert rms.diagnostics["method"] == "recursive_modular_shell_rmsnorm_only"
    assert rms.diagnostics["base_method"] == "recursive_modular_shell"
    assert set(rms.diagnostics["rmsnorm_target_roles"]) == {"input_norm", "post_attention_norm"}


def test_effective_perturbation_on_target_is_halved_for_default_multiplier():
    named = dict(_named_parameters(num_layers=1))
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(list(named.items()), config)
    rms = build_recursive_modular_result_rmsnorm_only(list(named.items()), config)

    name = "model.layers.0.input_layernorm.weight"
    noise = torch.randn(8, generator=torch.Generator().manual_seed(11))
    radius = 0.08

    denom_v1 = normalization_denominator(
        method="recursive_modular_shell",
        name=name,
        noise=noise,
        modular_scale=v1.scales[name],
    )
    denom_rms = normalization_denominator(
        method="recursive_modular_shell_rmsnorm_only",
        name=name,
        noise=noise,
        modular_scale=rms.scales[name],
    )
    delta_v1 = radius * noise / denom_v1
    delta_rms = radius * noise / denom_rms
    assert math.isclose(
        natural_norm(name, delta_rms) / natural_norm(name, delta_v1),
        0.5,
        rel_tol=1e-9,
    )


def test_worker_extension_rejects_sensitivity_profile_for_rmsnorm_only():
    worker = WorkerExtension()
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    with pytest.raises(ValueError):
        worker._get_recursive_modular_result(
            named,
            config,
            method="recursive_modular_shell_rmsnorm_only",
            sensitivity_profile={"schema_version": 2},
        )


def test_worker_extension_builds_and_caches_rmsnorm_only_result():
    worker = WorkerExtension()
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    result = worker._get_recursive_modular_result(
        named,
        config,
        method="recursive_modular_shell_rmsnorm_only",
    )
    assert result.diagnostics["num_rmsnorm_parameters_scaled"] == 4
    cached = worker._get_recursive_modular_result(
        named,
        config,
        method="recursive_modular_shell_rmsnorm_only",
    )
    assert cached is result


def test_worker_extension_cache_key_distinguishes_multiplier():
    worker = WorkerExtension()
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    result_2x = worker._get_recursive_modular_result(
        named,
        config,
        method="recursive_modular_shell_rmsnorm_only",
        rmsnorm_multiplier=2.0,
    )
    result_3x = worker._get_recursive_modular_result(
        named,
        config,
        method="recursive_modular_shell_rmsnorm_only",
        rmsnorm_multiplier=3.0,
    )
    # A different multiplier must not silently reuse the 2x cache entry.
    assert result_2x is not result_3x
    name = "model.layers.0.input_layernorm.weight"
    assert not math.isclose(result_2x.scales[name], result_3x.scales[name])

    result_2x_again = worker._get_recursive_modular_result(
        named,
        config,
        method="recursive_modular_shell_rmsnorm_only",
        rmsnorm_multiplier=2.0,
    )
    assert result_2x_again is result_2x


def test_scale_statistics_reflect_doubled_scales_not_v1():
    named = _named_parameters(num_layers=3)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(named, config, rmsnorm_multiplier=2.0)

    # Global scale_statistics must be recomputed from the post-multiplier scales.
    assert rms.diagnostics["scale_statistics"] != v1.diagnostics["scale_statistics"]
    finite_scales = [v for v in rms.scales.values() if math.isfinite(v)]
    assert math.isclose(rms.diagnostics["scale_statistics"]["max"], max(finite_scales))
    assert math.isclose(rms.diagnostics["scale_statistics"]["min"], min(finite_scales))

    # The "norm" group summary must reflect the doubled RMSNorm scales too.
    norm_names = [
        name
        for name in rms.scales
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight"))
        or name == "model.norm.weight"
    ]
    expected_norm_summary_max = max(rms.scales[name] for name in norm_names)
    assert math.isclose(
        rms.diagnostics["group_statistics"]["norm"]["scale_summary"]["max"],
        expected_norm_summary_max,
    )
    # Non-norm groups (e.g. embedding) are untouched by this ablation, so their
    # summaries must still match V1 exactly.
    assert (
        rms.diagnostics["group_statistics"]["embedding"]
        == v1.diagnostics["group_statistics"]["embedding"]
    )


def test_non_target_delta_end_to_end_identical_to_v1():
    """Full apply-perturbation math (not just the scale value) must match V1
    exactly for every parameter outside the RMSNorm allowlist."""
    named = _named_parameters(num_layers=2)
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    rms = build_recursive_modular_result_rmsnorm_only(named, config)

    radius = 0.05
    for name, param in named:
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            continue
        noise = torch.randn(param.shape, generator=torch.Generator().manual_seed(99))
        denom_v1 = normalization_denominator(
            method="recursive_modular_shell", name=name, noise=noise, modular_scale=v1.scales[name]
        )
        denom_rms = normalization_denominator(
            method="recursive_modular_shell_rmsnorm_only",
            name=name,
            noise=noise,
            modular_scale=rms.scales[name],
        )
        assert math.isclose(denom_v1, denom_rms, rel_tol=1e-12)
        delta_v1 = radius * noise / denom_v1
        delta_rms = radius * noise / denom_rms
        assert torch.equal(delta_v1, delta_rms), name


class _FakeModel:
    def __init__(self, named_parameters):
        self._named = named_parameters

    def named_parameters(self):
        return iter(self._named)


class _FakeModelRunner:
    def __init__(self, named_parameters):
        self.model = _FakeModel(named_parameters)


def test_apply_then_restore_roundtrips_exactly():
    worker = WorkerExtension()
    named = _named_parameters(num_layers=2)
    worker.model_runner = _FakeModelRunner(named)
    originals = {name: param.data.clone() for name, param in named}
    config = load_mass_config(None)

    worker._apply_weight_perturbation(
        seed=123,
        radius=0.05,
        method="recursive_modular_shell_rmsnorm_only",
        mass_config=config,
        restore=False,
    )
    assert any(not torch.equal(param.data, originals[name]) for name, param in named)

    worker._apply_weight_perturbation(
        seed=123,
        radius=0.05,
        method="recursive_modular_shell_rmsnorm_only",
        mass_config=config,
        restore=True,
    )
    for name, param in named:
        assert torch.allclose(param.data, originals[name], atol=1e-6), name
