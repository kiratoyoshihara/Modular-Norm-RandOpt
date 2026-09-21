from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import torch

from utils.sensitivity_calibration import first_tensor


@dataclass(frozen=True)
class MetricSpec:
    name: str
    function: Callable[[torch.Tensor], torch.Tensor]
    point: torch.Tensor


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


def extract_hidden(capture: Mapping[str, Any]) -> torch.Tensor:
    kwargs = capture["kwargs"]
    value = kwargs.get("hidden_states")
    if isinstance(value, torch.Tensor):
        return value
    for item in capture["args"]:
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            return item
    raise RuntimeError("Could not identify hidden_states in captured module inputs")


def attention_kwargs(capture: Mapping[str, Any]) -> Dict[str, Any]:
    clean = dict(detach_tree(capture["kwargs"]))
    clean.pop("hidden_states", None)
    clean.pop("past_key_value", None)
    clean.pop("past_key_values", None)
    clean["use_cache"] = False
    clean["output_attentions"] = False
    return clean


def call_attention(module, hidden: torch.Tensor, kwargs: Mapping[str, Any]):
    call_kwargs = dict(kwargs)
    try:
        return first_tensor(module(hidden_states=hidden, **call_kwargs))
    except TypeError:
        call_kwargs.pop("use_cache", None)
        call_kwargs.pop("output_attentions", None)
        try:
            return first_tensor(module(hidden_states=hidden, **call_kwargs))
        except TypeError:
            return first_tensor(module(hidden, **call_kwargs))


class ArchitectureAdapter:
    family = ""
    supported_model_types: tuple[str, ...] = ()
    block_layout = ""
    required_metrics: tuple[str, ...] = ()

    def get_decoder(self, model):
        for candidate in (
            getattr(model, "model", None),
            getattr(model, "transformer", None),
            getattr(model, "base_model", None),
            model,
        ):
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate
        raise RuntimeError("Could not locate decoder layers on the loaded model")

    def validate_layer(self, layer, layer_index: int) -> None:
        raise NotImplementedError

    def capture_modules(self, layer) -> Mapping[str, torch.nn.Module]:
        raise NotImplementedError

    def metric_specs(self, layer, captures, layer_index: int) -> Sequence[MetricSpec]:
        raise NotImplementedError

    def layer_metadata(self, layer) -> Mapping[str, Any]:
        return {}
