from __future__ import annotations

from .base import (
    ArchitectureAdapter,
    MetricSpec,
    attention_kwargs,
    call_attention,
    extract_hidden,
)


class Qwen2Adapter(ArchitectureAdapter):
    family = "qwen2"
    supported_model_types = ("qwen2", "qwen2_moe")
    block_layout = "pre_norm"
    required_metrics = (
        "input_norm",
        "post_attention_norm",
        "attention_module",
        "attention_residual",
        "mlp_module",
        "mlp_residual",
        "block",
    )

    def validate_layer(self, layer, layer_index: int) -> None:
        for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            if not hasattr(layer, name):
                raise RuntimeError(f"Layer {layer_index} lacks required attribute {name}")

    def capture_modules(self, layer):
        return {
            "layer": layer,
            "self_attn": layer.self_attn,
            "post_attention_norm": layer.post_attention_layernorm,
            "mlp": layer.mlp,
        }

    def metric_specs(self, layer, captures, layer_index):
        block_input = extract_hidden(captures["layer"])
        attn_input = extract_hidden(captures["self_attn"])
        residual_stream = extract_hidden(captures["post_attention_norm"])
        mlp_input = extract_hidden(captures["mlp"])
        kwargs = attention_kwargs(captures["self_attn"])
        attention = lambda h: call_attention(layer.self_attn, h, kwargs)
        attention_residual = lambda h: h + attention(layer.input_layernorm(h))
        mlp = lambda h: layer.mlp(h)
        mlp_residual = lambda h: h + mlp(layer.post_attention_layernorm(h))
        block = lambda h: mlp_residual(attention_residual(h))
        return (
            MetricSpec("input_norm", layer.input_layernorm, block_input),
            MetricSpec("post_attention_norm", layer.post_attention_layernorm, residual_stream),
            MetricSpec("attention_module", attention, attn_input),
            MetricSpec("attention_residual", attention_residual, block_input),
            MetricSpec("mlp_module", mlp, mlp_input),
            MetricSpec("mlp_residual", mlp_residual, residual_stream),
            MetricSpec("block", block, block_input),
        )
