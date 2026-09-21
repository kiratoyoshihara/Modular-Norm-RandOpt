"""Helpers for models whose public config wraps a decoder text config."""

from __future__ import annotations


def effective_text_config(config):
    """Return the decoder config used by perturbation/calibration code."""

    return getattr(config, "text_config", config)


def transformers_causal_model_class(config):
    """Choose the HF auto class whose text logits match the vLLM model."""

    if getattr(config, "model_type", None) == "gemma3":
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


__all__ = ["effective_text_config", "transformers_causal_model_class"]
