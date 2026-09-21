"""Source fingerprints for the local official-RandOpt + J(r) protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SOURCE_PATHS = (
    "population_scaling.py",
    "core/__init__.py",
    "core/engine.py",
    "data_handlers/__init__.py",
    "data_handlers/base.py",
    "data_handlers/countdown.py",
    "data_handlers/gsm8k.py",
    "scripts/measure_functional_displacement.py",
    "utils/functional_displacement.py",
    "utils/distance_match_artifact.py",
    "utils/distance_matching.py",
    "utils/hf_vllm_parameter_bridge.py",
    "utils/model_config.py",
    "utils/official_randopt_protocol.py",
    "utils/official_prompt_protocol.py",
    "utils/official_randopt_provenance.py",
    "utils/olmo3_compat.py",
    "utils/perturbation_norms.py",
    "utils/recursive_modular_v1.py",
    "utils/recursive_modular_v2.py",
    "utils/worker_extn.py",
    "utils/architecture_adapters/__init__.py",
    "utils/architecture_adapters/base.py",
    "utils/architecture_adapters/gemma3.py",
    "utils/architecture_adapters/llama.py",
    "utils/architecture_adapters/olmo3.py",
    "utils/architecture_adapters/qwen2.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_manifest(repo_root: str | Path) -> Mapping[str, Any]:
    root = Path(repo_root)
    files = {
        path: sha256_bytes((root / path).read_bytes())
        for path in SOURCE_PATHS
    }
    combined = sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {"combined_sha256": combined, "files": files}


__all__ = ["SOURCE_PATHS", "sha256_bytes", "source_manifest"]
