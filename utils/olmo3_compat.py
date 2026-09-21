"""Compatibility helpers for OLMo3 configs consumed by vLLM."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_vllm_olmo3_hf_overrides(config: Any) -> Dict[str, Any]:
    """Flatten Transformers' per-attention-type OLMo3 RoPE configuration.

    Transformers 5 represents OLMo3 RoPE parameters as a mapping with
    sliding_attention and full_attention rows. vLLM 0.19's shared OLMo2/3
    runtime expects the full-attention YaRN dictionary directly and extracts
    rope_theta from it for sliding-attention layers.
    """

    if getattr(config, "model_type", None) != "olmo3":
        return {}
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, Mapping):
        return {}
    full_attention = rope_parameters.get("full_attention")
    if not isinstance(full_attention, Mapping):
        return {}

    flattened = dict(full_attention)
    if "rope_theta" not in flattened:
        sliding_attention = rope_parameters.get("sliding_attention", {})
        if isinstance(sliding_attention, Mapping):
            rope_theta = sliding_attention.get("rope_theta")
            if rope_theta is not None:
                flattened["rope_theta"] = rope_theta
    if "rope_theta" not in flattened:
        raise ValueError(
            "OLMo3 RoPE configuration has no rope_theta in either full or "
            "sliding attention settings"
        )
    return {"rope_parameters": flattened}
