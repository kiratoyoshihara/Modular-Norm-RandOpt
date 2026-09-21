"""OLMo3 TP=1 vocabulary padding support without changing the shared runtime.

vLLM rounds the embedding/head row count up to a multiple of 64. Generate and
normalize noise over that full physical shape, then apply only the checkpoint
rows to Transformers. Padded tokens are neither inputs nor output classes.
The model, vocabulary, radius, masses, noise rule and restoration are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from utils.hf_vllm_parameter_bridge import (
    ParameterFragment, build_physical_parameter_bindings as base_bindings,
)
from utils.official_randopt_provenance import source_manifest as base_source_manifest

COMPAT_SOURCE_PATHS = (
    "utils/olmo3_padded_vocab_bridge.py",
    "scripts/functional_displacement/measure_olmo3_functional_displacement.py",
)


def build_physical_parameter_bindings(model, runtime_rows):
    if getattr(model.config, "model_type", None) != "olmo3":
        raise ValueError("The padded-vocabulary adapter is restricted to OLMo3")
    parameters = dict(model.named_parameters())
    vocabulary = int(model.config.vocab_size)
    allowed = {"model.embed_tokens.weight": "embedding", "lm_head.weight": "lm_head"}
    adjusted, padded = [], {}
    for row in runtime_rows:
        row = dict(row)
        name = row["parameter_name"]
        shape = tuple(row["shape"])
        if name in allowed and name in parameters and shape != tuple(parameters[name].shape):
            checkpoint_shape = tuple(parameters[name].shape)
            if (row.get("role") != allowed[name] or len(shape) != 2 or len(checkpoint_shape) != 2
                    or checkpoint_shape[0] != vocabulary or shape[1] != checkpoint_shape[1]
                    or shape[0] != ((vocabulary + 63) // 64) * 64 or shape[0] <= vocabulary):
                raise ValueError(f"Not supported OLMo3 TP=1 vocabulary padding: {name}")
            padded[name] = shape
            row["shape"] = list(checkpoint_shape)
        adjusted.append(row)
    bindings, parameters = base_bindings(model, adjusted)
    result = []
    for binding in bindings:
        if binding.physical_name in padded:
            fragment = binding.fragments[0]
            binding = replace(binding, shape=padded[binding.physical_name], fragments=(
                ParameterFragment(fragment.name, start=0, stop=vocabulary),))
        result.append(binding)
    return result, parameters


def source_manifest(repo_root):
    """Fingerprint the unchanged shared runtime AND this launch-local adapter."""
    root = Path(repo_root)
    files = dict(base_source_manifest(root)["files"])
    files.update({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in COMPAT_SOURCE_PATHS})
    combined = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"combined_sha256": combined, "files": files}


def load_and_validate_radius_artifact(args):
    """Apply the standard artifact checks including this adapter's provenance.

    A later OLMo production launcher must use this same source-manifest provider
    to validate an artifact produced by this isolated compatibility entry point.
    """
    import population_scaling
    original = population_scaling.source_manifest
    population_scaling.source_manifest = source_manifest
    try:
        return population_scaling.load_and_validate_radius_artifact(args)
    finally:
        population_scaling.source_manifest = original
