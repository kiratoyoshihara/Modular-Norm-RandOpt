from __future__ import annotations

from .base import ArchitectureAdapter, MetricSpec, attention_kwargs, call_attention, extract_hidden


class Gemma3Adapter(ArchitectureAdapter):
    """Gemma 3 decoder with pre-norm inputs and post-branch norms."""

    family = "gemma3"
    supported_model_types = ("gemma3", "gemma3_text")
    block_layout = "pre_norm_branch_post_norm"
    required_metrics = (
        "input_norm", "q_norm", "k_norm", "attention_module",
        "attention_output_norm", "attention_residual", "pre_feedforward_norm",
        "mlp_module", "mlp_output_norm", "mlp_residual", "block",
    )

    def get_decoder(self, model):
        for candidate in (
            getattr(getattr(model, "model", None), "language_model", None),
            getattr(model, "language_model", None),
            getattr(model, "model", None), model,
        ):
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate
        raise RuntimeError("Could not locate Gemma 3 language-model decoder layers")

    def validate_layer(self, layer, layer_index: int) -> None:
        for name in (
            "input_layernorm", "self_attn", "post_attention_layernorm",
            "pre_feedforward_layernorm", "mlp", "post_feedforward_layernorm",
        ):
            if not hasattr(layer, name):
                raise RuntimeError(f"Layer {layer_index} lacks required attribute {name}")
        for name in ("q_norm", "k_norm"):
            if not hasattr(layer.self_attn, name):
                raise RuntimeError(f"Layer {layer_index}.self_attn lacks {name}")

    def capture_modules(self, layer):
        return {
            "layer": layer,
            "input_norm": layer.input_layernorm,
            "self_attn": layer.self_attn,
            "q_norm": layer.self_attn.q_norm,
            "k_norm": layer.self_attn.k_norm,
            "attention_output_norm": layer.post_attention_layernorm,
            "pre_feedforward_norm": layer.pre_feedforward_layernorm,
            "mlp": layer.mlp,
            "mlp_output_norm": layer.post_feedforward_layernorm,
        }

    def metric_specs(self, layer, captures, layer_index):
        block_input = extract_hidden(captures["layer"])
        input_norm_input = extract_hidden(captures["input_norm"])
        attention_input = extract_hidden(captures["self_attn"])
        attention_output = extract_hidden(captures["attention_output_norm"])
        pre_ff_input = extract_hidden(captures["pre_feedforward_norm"])
        mlp_input = extract_hidden(captures["mlp"])
        mlp_output = extract_hidden(captures["mlp_output_norm"])
        q_input = extract_hidden(captures["q_norm"])
        k_input = extract_hidden(captures["k_norm"])
        kwargs = attention_kwargs(captures["self_attn"])
        attention = lambda h: call_attention(layer.self_attn, layer.input_layernorm(h), kwargs)
        attention_branch = lambda h: layer.post_attention_layernorm(attention(h))
        attention_residual = lambda h: h + attention_branch(h)
        mlp = lambda h: layer.mlp(layer.pre_feedforward_layernorm(h))
        mlp_branch = lambda h: layer.post_feedforward_layernorm(mlp(h))
        mlp_residual = lambda h: h + mlp_branch(h)
        block = lambda h: mlp_residual(attention_residual(h))
        return (
            MetricSpec("input_norm", layer.input_layernorm, input_norm_input),
            MetricSpec("q_norm", layer.self_attn.q_norm, q_input),
            MetricSpec("k_norm", layer.self_attn.k_norm, k_input),
            MetricSpec("attention_module", attention, attention_input),
            MetricSpec("attention_output_norm", layer.post_attention_layernorm, attention_output),
            MetricSpec("attention_residual", attention_residual, block_input),
            MetricSpec("pre_feedforward_norm", layer.pre_feedforward_layernorm, pre_ff_input),
            MetricSpec("mlp_module", mlp, mlp_input),
            MetricSpec("mlp_output_norm", layer.post_feedforward_layernorm, mlp_output),
            MetricSpec("mlp_residual", mlp_residual, pre_ff_input),
            MetricSpec("block", block, block_input),
        )

    def layer_metadata(self, layer):
        return {"attention_type": getattr(layer, "attention_type", "unknown")}
