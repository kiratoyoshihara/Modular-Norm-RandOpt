"""Architecture adapters shared by sensitivity calibration and runtime metadata."""

from .base import ArchitectureAdapter, MetricSpec
from .gemma3 import Gemma3Adapter
from .llama import LlamaAdapter
from .olmo3 import Olmo3Adapter
from .qwen2 import Qwen2Adapter

_ADAPTERS = {
    "qwen2": Qwen2Adapter(),
    "llama": LlamaAdapter(),
    "olmo3": Olmo3Adapter(),
    "gemma3": Gemma3Adapter(),
}

_MODEL_TYPE_TO_FAMILY = {
    model_type: family
    for family, adapter in _ADAPTERS.items()
    for model_type in adapter.supported_model_types
}


def get_architecture_adapter(architecture: str, model_type: str | None = None):
    family = architecture
    if family == "auto":
        if not model_type:
            raise ValueError("model_type is required when architecture='auto'")
        family = _MODEL_TYPE_TO_FAMILY.get(model_type, "")
    if family not in _ADAPTERS:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unsupported architecture {architecture!r} (model_type={model_type!r}); "
            f"supported families: {supported}"
        )
    adapter = _ADAPTERS[family]
    if model_type and model_type not in adapter.supported_model_types:
        raise ValueError(
            f"Architecture {family!r} does not support model_type {model_type!r}"
        )
    return adapter


__all__ = [
    "ArchitectureAdapter",
    "MetricSpec",
    "get_architecture_adapter",
]
