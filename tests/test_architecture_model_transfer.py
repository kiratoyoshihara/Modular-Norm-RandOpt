from __future__ import annotations

import math

import pytest
import torch

from utils.architecture_adapters import get_architecture_adapter
from utils.olmo3_compat import build_vllm_olmo3_hf_overrides
from utils.perturbation_norms import load_mass_config
from utils.recursive_modular_v1 import infer_parameter_spec
from utils.recursive_modular_v2 import (
    build_recursive_modular_result_v2,
    load_sensitivity_profile,
)
from scripts.calibrate_decoder_sensitivities import _capture_inputs


def _parameter(shape, seed):
    return torch.nn.Parameter(
        torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    )


def _olmo3_named_parameters(num_layers=2):
    rows = [("model.embed_tokens.weight", _parameter((32, 8), 1))]
    seed = 2
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        entries = [
            (f"{prefix}.self_attn.q_proj.weight", (8, 8)),
            (f"{prefix}.self_attn.q_norm.weight", (8,)),
            (f"{prefix}.self_attn.k_proj.weight", (4, 8)),
            (f"{prefix}.self_attn.k_norm.weight", (4,)),
            (f"{prefix}.self_attn.v_proj.weight", (4, 8)),
            (f"{prefix}.self_attn.o_proj.weight", (8, 8)),
            (f"{prefix}.post_attention_layernorm.weight", (8,)),
            (f"{prefix}.mlp.gate_proj.weight", (16, 8)),
            (f"{prefix}.mlp.up_proj.weight", (16, 8)),
            (f"{prefix}.mlp.down_proj.weight", (8, 16)),
            (f"{prefix}.post_feedforward_layernorm.weight", (8,)),
        ]
        for name, shape in entries:
            rows.append((name, _parameter(shape, seed)))
            seed += 1
    rows.append(("model.norm.weight", _parameter((8,), seed)))
    rows.append(("lm_head.weight", _parameter((32, 8), seed + 1)))
    return rows


def _v3_profile(family, metrics, num_layers=2):
    layers = {}
    for layer in range(num_layers):
        layers[str(layer)] = {
            "metadata": {
                "attention_type": (
                    "sliding_attention" if layer % 2 == 0 else "full_attention"
                )
            },
            "metrics": {metric: {"value": 1.0} for metric in metrics},
        }
    return {
        "schema_version": 3,
        "method": "recursive_modular_shell_v2",
        "model_name": f"dummy-{family}",
        "architecture": {
            "family": family,
            "model_type": family,
            "block_layout": (
                "branch_post_norm" if family == "olmo3" else "pre_norm"
            ),
        },
        "application": {
            "linear_policy": "unit",
            "strict_profile": True,
            "propagation_mode": "local_only",
            "local_normalization": "per_block_median",
            "local_ratio_clip_min": 0.5,
            "local_ratio_clip_max": 2.0,
        },
        "global": {
            "final_norm": {"value": 1.0},
            "output_head": {"value": 1.0},
        },
        "layers": layers,
    }


def test_adapter_registry_auto_selects_all_supported_families():
    assert get_architecture_adapter("auto", "qwen2").family == "qwen2"
    assert get_architecture_adapter("auto", "llama").family == "llama"
    assert get_architecture_adapter("auto", "olmo3").family == "olmo3"


def test_schema_v3_loader_accepts_metadata_and_nested_metrics():
    profile = _v3_profile(
        "olmo3",
        get_architecture_adapter("olmo3", "olmo3").required_metrics,
    )
    loaded = load_sensitivity_profile(profile)
    assert loaded["layers"]["0"]["metadata"]["attention_type"] == "sliding_attention"


def test_olmo3_parameter_roles_are_complete_and_unique():
    named = _olmo3_named_parameters()
    specs = [infer_parameter_spec(name, param.shape, "olmo3") for name, param in named]
    assert all(spec.group != "other" for spec in specs)
    roles = {spec.name: spec.role for spec in specs}
    assert roles["model.layers.0.self_attn.q_norm.weight"] == "q_norm"
    assert roles["model.layers.0.self_attn.k_norm.weight"] == "k_norm"
    assert (
        roles["model.layers.0.post_attention_layernorm.weight"]
        == "attention_output_norm"
    )
    assert (
        roles["model.layers.0.post_feedforward_layernorm.weight"]
        == "mlp_output_norm"
    )


def test_olmo3_v3_shadow_tree_has_finite_bounded_scales():
    adapter = get_architecture_adapter("olmo3", "olmo3")
    profile = _v3_profile("olmo3", adapter.required_metrics)
    profile["layers"]["0"]["metrics"]["q_norm"]["value"] = 20.0
    profile["layers"]["0"]["metrics"]["mlp_output_norm"]["value"] = 0.05
    result = build_recursive_modular_result_v2(
        _olmo3_named_parameters(),
        load_mass_config(None),
        profile,
        power_iterations=8,
    )
    active = [
        row for row in result.diagnostics["parameters"] if row["assigned_mass"] > 0.0
    ]
    assert active
    assert all(math.isfinite(float(row["scale"])) for row in active)
    ratios = [float(row["final_v2_to_v1_ratio"]) for row in active]
    assert min(ratios) >= 0.5
    assert max(ratios) <= 2.0
    assert result.diagnostics["architecture_family"] == "olmo3"
    tree_text = str(result.diagnostics["tree"])
    assert "qk_norms" in tree_text
    assert "attention_output_norm" in tree_text
    assert "mlp_output_norm" in tree_text
    assert "structural_fallback_parameters" not in tree_text


def test_hf_llama_and_olmo3_adapters_execute_metric_graphs():
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        Olmo3Config,
        Olmo3ForCausalLM,
    )

    cases = (
        (
            "llama",
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                )
            ),
        ),
        (
            "olmo3",
            Olmo3ForCausalLM(
                Olmo3Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=32,
                    sliding_window=16,
                )
            ),
        ),
    )
    input_ids = torch.tensor([[1, 2, 3, 4]])
    for family, model in cases:
        adapter = get_architecture_adapter(family, model.config.model_type)
        decoder = adapter.get_decoder(model)
        layers = list(decoder.layers)
        with _capture_inputs(model, decoder, layers, adapter) as captures:
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
        prefix = "layer.0."
        local = {
            key.removeprefix(prefix): value
            for key, value in captures.items()
            if key.startswith(prefix)
        }
        specs = adapter.metric_specs(layers[0], local, 0)
        assert tuple(spec.name for spec in specs) == adapter.required_metrics
        for spec in specs:
            with torch.no_grad():
                output = spec.function(spec.point)
            assert output.shape == spec.point.shape
            assert torch.isfinite(output).all()


def test_schema_v3_rejects_unclassified_runtime_parameters():
    adapter = get_architecture_adapter("olmo3", "olmo3")
    profile = _v3_profile("olmo3", adapter.required_metrics)
    named = _olmo3_named_parameters()
    named.append(("model.layers.0.mystery.weight", _parameter((8,), 99)))
    with pytest.raises(ValueError, match="unclassified"):
        build_recursive_modular_result_v2(
            named,
            load_mass_config(None),
            profile,
            power_iterations=4,
        )


def test_olmo3_vllm_rope_override_flattens_full_attention_config():
    class Config:
        model_type = "olmo3"
        rope_parameters = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000.0,
            },
            "full_attention": {
                "rope_type": "yarn",
                "factor": 8.0,
                "rope_theta": 500000.0,
            },
        }

    overrides = build_vllm_olmo3_hf_overrides(Config())
    assert overrides == {
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 8.0,
            "rope_theta": 500000.0,
        }
    }


def test_vllm_rope_override_is_empty_for_llama():
    class Config:
        model_type = "llama"
        rope_parameters = {"rope_type": "llama3"}

    assert build_vllm_olmo3_hf_overrides(Config()) == {}
