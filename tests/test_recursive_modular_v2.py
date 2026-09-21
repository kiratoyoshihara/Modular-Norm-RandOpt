from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.perturbation_norms import (  # noqa: E402
    load_mass_config,
    natural_norm,
    normalization_denominator,
)
from utils.recursive_modular_v1 import build_recursive_modular_result  # noqa: E402
from utils.recursive_modular_v2 import (  # noqa: E402
    build_recursive_modular_result_v2,
    load_sensitivity_profile,
    sensitivity_profile_fingerprint,
)
from utils.sensitivity_calibration import (  # noqa: E402
    aggregate_log_quantile,
    estimate_operator_norm_power,
)


def _parameter(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape, generator=generator))


def _named_parameters(num_layers=2, include_head=True):
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


def _profile(num_layers=2, **application_overrides):
    application = {
        "clip_min": 0.01,
        "clip_max": 100.0,
        "factor_clip_min": 0.01,
        "factor_clip_max": 100.0,
        "linear_policy": "unit",
        "strict_profile": True,
        "propagation_mode": "local_only",
        "local_normalization": "per_block_median",
        "local_ratio_clip_min": 0.5,
        "local_ratio_clip_max": 2.0,
        # Legacy fields intentionally retained to test backward compatibility.
        "scale_normalization": "match_v1_median",
        "block_reconciliation": "geometric",
    }
    application.update(application_overrides)
    layers = {}
    for layer in range(num_layers):
        layers[str(layer)] = {
            "input_norm": {"value": 1.0},
            "post_attention_norm": {"value": 1.0},
            "attention_module": {"value": 1.0},
            "attention_residual": {"value": 1.0},
            "mlp_module": {"value": 1.0},
            "mlp_residual": {"value": 1.0},
            "block": {"value": 1.0},
        }
    return {
        "schema_version": 2,
        "model_name": "dummy-qwen",
        "application": application,
        "global": {"final_norm": {"value": 1.0}, "output_head": {"value": 1.0}},
        "layers": layers,
    }


def _active_ratio_map(v1, v2):
    return {
        name: v2.scales[name] / v1.scales[name]
        for name, mass in v2.assigned_masses.items()
        if mass > 0.0
    }


def test_unit_profile_reproduces_v1_scales():
    named = _named_parameters()
    config = load_mass_config(None)
    v1 = build_recursive_modular_result(named, config)
    v2 = build_recursive_modular_result_v2(named, config, _profile(), power_iterations=20)
    assert set(v1.scales) == set(v2.scales)
    for name in v1.scales:
        assert math.isclose(v1.scales[name], v2.scales[name], rel_tol=1e-10)


def test_extreme_block_and_residual_metrics_are_diagnostic_only():
    named = _named_parameters()
    config = load_mass_config(None)
    base = build_recursive_modular_result_v2(named, config, _profile(), power_iterations=10)
    extreme_profile = _profile()
    for row in extreme_profile["layers"].values():
        row["attention_residual"]["value"] = 1e9
        row["mlp_residual"]["value"] = 1e8
        row["block"]["value"] = 1e12
    changed = build_recursive_modular_result_v2(
        named, config, extreme_profile, power_iterations=10
    )
    for name in base.scales:
        assert math.isclose(base.scales[name], changed.scales[name], rel_tol=1e-12)
    assert changed.diagnostics["safety_checks"]["no_depthwise_sensitivity_product"]
    assert changed.diagnostics["root_sensitivity"] == 1.0


def test_output_head_sensitivity_does_not_scale_decoder_layers():
    named = _named_parameters(include_head=False)
    config = load_mass_config(None)
    base_profile = _profile()
    base = build_recursive_modular_result_v2(named, config, base_profile, power_iterations=10)
    changed_profile = _profile()
    changed_profile["global"]["output_head"]["value"] = 1e6
    changed = build_recursive_modular_result_v2(
        named, config, changed_profile, power_iterations=10
    )
    for name in base.scales:
        assert math.isclose(base.scales[name], changed.scales[name], rel_tol=1e-12)


def test_local_ratios_are_bounded_and_each_block_is_centered():
    named = _named_parameters()
    config = load_mass_config(None)
    profile = _profile(linear_policy="analytic", clip_min=0.01, clip_max=100.0)
    profile["layers"]["0"]["input_norm"]["value"] = 500.0
    profile["layers"]["0"]["attention_module"]["value"] = 1000.0
    profile["layers"]["0"]["mlp_module"]["value"] = 0.001
    profile["layers"]["1"]["post_attention_norm"]["value"] = 0.001
    result = build_recursive_modular_result_v2(named, config, profile, power_iterations=30)
    v1 = build_recursive_modular_result(named, config)
    ratios = _active_ratio_map(v1, result)
    assert min(ratios.values()) >= 0.5 - 1e-12
    assert max(ratios.values()) <= 2.0 + 1e-12
    for layer in (0, 1):
        layer_ratios = sorted(
            ratio
            for name, ratio in ratios.items()
            if f".layers.{layer}." in name
        )
        assert layer_ratios
        median = torch.median(torch.tensor(layer_ratios)).item()
        assert math.isclose(median, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert result.diagnostics["safety_checks"]["final_ratio_within_bounds"]


def test_non_layer_parameters_remain_exactly_v1():
    named = _named_parameters()
    config = load_mass_config(None)
    profile = _profile(linear_policy="analytic")
    for row in profile["layers"].values():
        row["attention_module"]["value"] = 20.0
        row["mlp_module"]["value"] = 0.2
    result = build_recursive_modular_result_v2(named, config, profile, power_iterations=20)
    v1 = build_recursive_modular_result(named, config)
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        assert math.isclose(result.scales[name], v1.scales[name], rel_tol=1e-12)


def test_legacy_schema_v2_profile_gets_safe_defaults():
    profile = _profile()
    for key in (
        "propagation_mode",
        "local_normalization",
        "local_ratio_clip_min",
        "local_ratio_clip_max",
    ):
        profile["application"].pop(key)
    loaded = load_sensitivity_profile(profile)
    result = build_recursive_modular_result_v2(
        _named_parameters(), load_mass_config(None), loaded, power_iterations=10
    )
    app = result.diagnostics["application"]
    assert app["propagation_mode"] == "local_only"
    assert app["local_ratio_clip_min"] == 0.5
    assert app["local_ratio_clip_max"] == 2.0


def test_profile_loader_and_fingerprint_are_deterministic():
    a = load_sensitivity_profile(_profile())
    b = load_sensitivity_profile(_profile())
    assert sensitivity_profile_fingerprint(a) == sensitivity_profile_fingerprint(b)


def test_profile_loader_accepts_pathlib_path(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    assert load_sensitivity_profile(path) == load_sensitivity_profile(_profile())


def test_power_estimator_matches_known_linear_operator():
    matrix = torch.diag(torch.tensor([3.0, 2.0, 0.5]))
    point = torch.randn(4, 3)
    value = estimate_operator_norm_power(lambda x: x @ matrix.T, point, power_iterations=12)
    assert math.isclose(value, 3.0, rel_tol=1e-4, abs_tol=1e-4)


def test_log_quantile_and_v2_shell_radius():
    assert math.isclose(aggregate_log_quantile([1.0, 2.0, 4.0], 0.5), 2.0, rel_tol=1e-7)
    name = "model.layers.0.self_attn.o_proj.weight"
    noise = torch.randn(12, 8, generator=torch.Generator().manual_seed(7))
    scale = 17.0
    radius = 0.08
    denominator = normalization_denominator(
        method="recursive_modular_shell_v2",
        name=name,
        noise=noise,
        modular_scale=scale,
        power_iterations=20,
    )
    delta = radius * noise / denominator
    achieved = scale * natural_norm(name, delta, power_iterations=20)
    assert math.isclose(achieved, radius, rel_tol=2e-4, abs_tol=2e-5)
