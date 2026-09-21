from .qwen2 import Qwen2Adapter


class LlamaAdapter(Qwen2Adapter):
    family = "llama"
    supported_model_types = ("llama",)
