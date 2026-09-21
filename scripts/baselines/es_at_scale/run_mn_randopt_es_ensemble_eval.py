#!/usr/bin/env python3
"""Evaluate K=10/25 MN-RandOpt ensembles from saved N=3000 searches.

This runner never performs candidate search.  It validates one completed
N=3000/K=1 ES-final MN-RandOpt run, reconstructs the top-25 experts for two
nested population prefixes, reuses the saved K=1 held-out prediction, and
runs held-out inference only for experts whose predictions are missing.

The four reported endpoints are fixed by task:

* Countdown: N in {2820, 3000}, K in {10, 25}
* GSM8K:     N in {2841, 3000}, K in {10, 25}

N=2820 and N=2841 are the non-exceeding total-model-prompt-evaluation matches
to the corresponding ES-at-Scale N=3000/K=1 rows when K=25.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

import population_scaling as population_runner  # noqa: E402
from core import cleanup_engines, launch_engines  # noqa: E402
from data_handlers import get_dataset_handler  # noqa: E402
from utils.official_prompt_protocol import prompt_manifest  # noqa: E402
from utils.official_randopt_protocol import build_candidate_seeds  # noqa: E402
from utils.perturbation_norms import (  # noqa: E402
    load_sensitivity_profile,
    sensitivity_profile_fingerprint,
)


PROTOCOL = "mn-randopt-es-at-scale-ensemble-eval-v1"
SOURCE_PROTOCOL = "mn-randopt-es-at-scale-final-v1"
SOURCE_POPULATION_SIZE = 3_000
TOP_K_VALUES = (10, 25)
TRAIN_SAMPLES = 200
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CHAT_TEMPLATE_DATE = "11 Aug 2026"
PERTURBATION_METHOD = "recursive_modular_shell_v2"
RADIUS = 0.16
MAX_TOKENS = 1_024
PRECISION = "bfloat16"
POWER_ITERATIONS = 8
SEEDS = (42, 43, 44)
MASS_CONFIG = {
    "embedding": 1.0,
    "attention": 0.5,
    "mlp": 0.5,
    "head": 1.0,
    "norm": 0.1,
    "other": 0.1,
}
SENSITIVITY_PROFILE_FINGERPRINT = "a84d3cd9dc1a1f43"
SENSITIVITY_PROFILE_PATH = (
    REPO_ROOT / "profiles/qwen2.5-1.5b-countdown-modular-shell.json"
)
SENSITIVITY_PROFILE_SHA256 = (
    "da5397c1dc884b83c4c7531c2f1cd01af97ea7488f56aa24b1fb20ca7e22a1bf"
)
PROMPT_SCHEME = "chat-template-single-tokenization-fixed-date-v1"

TASK_CONFIG: Mapping[str, Mapping[str, Any]] = {
    "countdown": {
        "prefixes": (2_820, 3_000),
        "train_path": REPO_ROOT / "data/countdown/countdown_train.json",
        "test_path": REPO_ROOT / "data/countdown/countdown_test.json",
        "train_data_sha256": "86f24778f6b545ae9ddb4c3fce6d3e7ce66fd0786632d5df67ca559c7c16337c",
        "test_data_sha256": "c2f9a47a63fd077f78ff51fac29955916ad7b23e96e5cb0022294dd86632c8bd",
        "train_prompt_sha256": "a6c88ad9f1f627862ca9278da9ec9902070dd085d91709deb6d187570bc6dc52",
        "test_prompt_sha256": "8c1d070ad16ffa1871848e61cb4ef9d7dbca3a1d5ae3e2a233d5333866551c15",
        "train_prompt_tokens": 25_575,
        "test_prompt_tokens": 191_663,
        "test_samples": 1_500,
        "es_total_model_prompt_evaluations": 601_500,
    },
    "gsm8k": {
        "prefixes": (2_841, 3_000),
        "train_path": REPO_ROOT / "data/gsm8k/train_200.parquet",
        "test_path": REPO_ROOT / "data/gsm8k/test.parquet",
        "train_data_sha256": "4aff68b6180f627c84444e3384b7ed6ae1fcf082d8a456aeebd8022afd478205",
        "test_data_sha256": "09cb3b2cd84ec2c679d600e79c094c80606d31497e78fcdf4bc4ab787c92e91f",
        "train_prompt_sha256": "83f2ed0bae82b803f9e7606968ba95b8170c29bf20c80a0334930182f661d26d",
        "test_prompt_sha256": "68b53d1810e2c9813d28cf49fb83106caa32bd699e1e61b9a337c247d4dfcfdf",
        "train_prompt_tokens": 20_978,
        "test_prompt_tokens": 138_901,
        "test_samples": 1_319,
        "es_total_model_prompt_evaluations": 601_319,
    },
}


for _config in TASK_CONFIG.values():
    for _key in ("train_path", "test_path"):
        _config[_key] = Path(os.environ.get("MN_RANDOPT_DATA", REPO_ROOT / "data")) / Path(_config[_key]).relative_to(REPO_ROOT / "data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _require_equal(actual: Any, expected: Any, label: str, path: Path) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch in {path}: actual={actual!r}, expected={expected!r}"
        )


def _require_close(actual: Any, expected: float, label: str, path: Path) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric in {path}: {actual!r}") from exc
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"{label} mismatch in {path}: actual={value!r}, expected={expected!r}"
        )


def _validate_prompt_manifest(
    manifest: Mapping[str, Any],
    *,
    examples: int,
    tokens: int,
    digest: str,
    label: str,
    path: Path,
) -> None:
    expected = {
        "scheme": PROMPT_SCHEME,
        "tokenization_scheme": PROMPT_SCHEME,
        "chat_template_date": CHAT_TEMPLATE_DATE,
        "add_special_tokens_after_render": False,
        "double_bos_prompt_count": 0,
        "num_prompts": examples,
        "total_tokens": tokens,
        "token_ids_sha256": digest,
    }
    for key, wanted in expected.items():
        _require_equal(manifest.get(key), wanted, f"{label} prompt {key}", path)


def select_prefix_experts(
    records: Sequence[Mapping[str, Any]],
    prefixes: Sequence[int],
    *,
    max_k: int = max(TOP_K_VALUES),
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Reproduce the original stable reward/index ranking without inference."""

    if max_k <= 0:
        raise ValueError("max_k must be positive")
    selections: dict[int, list[dict[str, Any]]] = {}
    union_by_seed: dict[int, dict[str, Any]] = {}
    for prefix in prefixes:
        if prefix <= 0 or prefix > len(records):
            raise ValueError(f"Invalid population prefix {prefix} for {len(records)} records")
        ranked = sorted(
            records[:prefix],
            key=lambda row: (-float(row["train_reward"]), int(row["candidate_index"])),
        )
        selected: list[dict[str, Any]] = []
        for rank, raw in enumerate(ranked[: min(max_k, prefix)], 1):
            row = dict(raw)
            row["rank_within_prefix"] = rank
            selected.append(row)
            union_by_seed.setdefault(int(row["seed"]), row)
        selections[int(prefix)] = selected
    union = sorted(union_by_seed.values(), key=lambda row: int(row["candidate_index"]))
    return selections, union


def build_fresh_compute_budgets(task: str) -> list[dict[str, Any]]:
    """Return from-scratch evaluation counts; these are not incremental costs."""

    config = TASK_CONFIG[task]
    heldout_samples = int(config["test_samples"])
    es_total = int(config["es_total_model_prompt_evaluations"])
    rows: list[dict[str, Any]] = []
    for prefix in config["prefixes"]:
        for top_k in TOP_K_VALUES:
            search = int(prefix) * TRAIN_SAMPLES
            heldout = int(top_k) * heldout_samples
            total = search + heldout
            rows.append(
                {
                    "population_size": int(prefix),
                    "top_k": int(top_k),
                    "search_model_prompt_evaluations": search,
                    "heldout_model_prompt_evaluations": heldout,
                    "total_model_prompt_evaluations": total,
                    "es_reference_total_model_prompt_evaluations": es_total,
                    "difference_from_es_total": total - es_total,
                }
            )
    return rows


def plan_prediction_cache(
    selected_union: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Split selected experts into validated cache hits and missing experts."""

    wanted = {int(row["seed"]): dict(row) for row in selected_union}
    cached: dict[int, dict[str, Any]] = {}
    for raw in cached_rows:
        seed = int(raw.get("seed", -1))
        if seed not in wanted:
            continue
        if seed in cached:
            raise ValueError(f"Duplicate cached prediction for selected seed {seed}")
        selected = wanted[seed]
        if int(raw.get("candidate_index", -1)) != int(selected["candidate_index"]):
            raise ValueError(f"Cached candidate index mismatch for seed {seed}")
        _require_close(raw.get("radius"), RADIUS, "cached radius", Path("cache"))
        _require_close(
            raw.get("train_reward"),
            float(selected["train_reward"]),
            "cached train reward",
            Path("cache"),
        )
        cached[seed] = dict(raw)
    missing = [dict(row) for row in selected_union if int(row["seed"]) not in cached]
    return cached, missing


def _validate_fixed_local_inputs(task: str) -> None:
    config = TASK_CONFIG[task]
    paths = {
        "train data": (Path(config["train_path"]), str(config["train_data_sha256"])),
        "test data": (Path(config["test_path"]), str(config["test_data_sha256"])),
        "sensitivity profile": (
            SENSITIVITY_PROFILE_PATH,
            SENSITIVITY_PROFILE_SHA256,
        ),
    }
    for label, (path, expected) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing fixed {label}: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Fixed {label} SHA-256 mismatch: actual={actual}, expected={expected}"
            )


def validate_source_run(source_run_dir: Path) -> dict[str, Any]:
    """Fail closed unless source_run_dir is the completed fixed N=3000/K=1 run."""

    source_run_dir = source_run_dir.expanduser().resolve()
    results_path = source_run_dir / "results.json"
    args_path = source_run_dir / "args.json"
    manifest_path = source_run_dir / "matched_run_manifest.json"
    candidate_path = source_run_dir / "candidate_records.jsonl"
    expert_path = source_run_dir / "expert_predictions.jsonl"
    selection_path = source_run_dir / "selected_models_by_prefix.json"
    profile_path = source_run_dir / "sensitivity_profile.json"

    results = _load_json(results_path)
    source_args = _load_json(args_path)
    manifest = _load_json(manifest_path)
    candidates = _load_jsonl(candidate_path)
    cached_rows = _load_jsonl(expert_path)
    source_selection = _load_json(selection_path)
    source_profile = _load_json(profile_path)

    task = str(results.get("dataset"))
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported source task in {results_path}: {task!r}")
    seed = int(results.get("global_seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"Unexpected source seed in {results_path}: {seed}")
    config = TASK_CONFIG[task]
    _validate_fixed_local_inputs(task)

    fixed_results = {
        "status": "complete",
        "model": MODEL,
        "execution_protocol": "official_randopt_j_v1",
        "perturbation_method": PERTURBATION_METHOD,
        "mass_config": MASS_CONFIG,
        "sensitivity_profile_fingerprint": SENSITIVITY_PROFILE_FINGERPRINT,
        "power_iterations": POWER_ITERATIONS,
        "chat_template_date": CHAT_TEMPLATE_DATE,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": int(config["test_samples"]),
        "population_size": SOURCE_POPULATION_SIZE,
        "candidate_seed_pool_size": SOURCE_POPULATION_SIZE,
        "population_prefixes": [SOURCE_POPULATION_SIZE],
        "top_k_values": [1],
        "unique_validation_experts": 1,
        "base_validation_accuracy": None,
    }
    for key, wanted in fixed_results.items():
        _require_equal(results.get(key), wanted, f"source {key}", results_path)
    _require_close(results.get("radius"), RADIUS, "source radius", results_path)

    fixed_args = {
        "dataset": task,
        "model_name": MODEL,
        "model_revision": MODEL_REVISION,
        "precision": PRECISION,
        "max_tokens": MAX_TOKENS,
        "chat_template_date": CHAT_TEMPLATE_DATE,
        "perturbation_method": PERTURBATION_METHOD,
        "radius": RADIUS,
        "mass_config": MASS_CONFIG,
        "power_iterations": POWER_ITERATIONS,
        "population_size": SOURCE_POPULATION_SIZE,
        "population_prefixes": str(SOURCE_POPULATION_SIZE),
        "top_k_values": "1",
        "global_seed": seed,
        "train_samples": TRAIN_SAMPLES,
        "test_samples": None,
        "num_engines": 1,
        "tp": 1,
        "base_only": False,
        "wall_clock_mode": True,
    }
    for key, wanted in fixed_args.items():
        actual = source_args.get(key)
        if key == "radius":
            _require_close(actual, float(wanted), f"source arg {key}", args_path)
        else:
            _require_equal(actual, wanted, f"source arg {key}", args_path)
    cuda_devices = str(source_args.get("cuda_devices", ""))
    if not cuda_devices or "," in cuda_devices:
        raise ValueError(f"Source run did not use exactly one GPU: {args_path}")

    environment = results.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"Missing source environment mapping: {results_path}")
    environment_expected = {
        "model_revision": MODEL_REVISION,
        "dtype": PRECISION,
        "inference_backend": "vllm",
        "wall_clock_protocol": SOURCE_PROTOCOL,
        "num_visible_gpus": 1,
        "num_engines": 1,
        "tensor_parallel_size": 1,
        "train_data_sha256": config["train_data_sha256"],
        "test_data_sha256": config["test_data_sha256"],
    }
    for key, wanted in environment_expected.items():
        _require_equal(environment.get(key), wanted, f"source environment {key}", results_path)
    if environment.get("resolved_model_revision") not in (None, MODEL_REVISION):
        raise ValueError(f"Resolved source model revision mismatch: {results_path}")
    current_runtime = {
        "python": sys.version,
        "torch": torch.__version__,
        "ray": importlib.metadata.version("ray"),
        "vllm": importlib.metadata.version("vllm"),
        "transformers": importlib.metadata.version("transformers"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    for key, actual in current_runtime.items():
        _require_equal(
            actual,
            environment.get(key),
            f"current/source runtime {key}",
            results_path,
        )

    matched = results.get("matched_protocol")
    if not isinstance(matched, Mapping):
        raise ValueError(f"Missing matched protocol: {results_path}")
    matched_expected = {
        "protocol": SOURCE_PROTOCOL,
        "es_iterations": 100,
        "candidate_evaluations": SOURCE_POPULATION_SIZE,
        "model_prompt_evaluations": SOURCE_POPULATION_SIZE * TRAIN_SAMPLES,
        "single_model_endpoint": True,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "task": task,
        "seed": seed,
        "test_samples": int(config["test_samples"]),
        "population_size": SOURCE_POPULATION_SIZE,
        "top_k": 1,
    }
    for key, wanted in matched_expected.items():
        _require_equal(matched.get(key), wanted, f"source matched protocol {key}", results_path)
    _require_equal(manifest.get("status"), "complete", "source manifest status", manifest_path)
    manifest_protocol = manifest.get("matched_protocol", {})
    if not isinstance(manifest_protocol, Mapping):
        raise ValueError(f"Invalid source matched manifest: {manifest_path}")
    _require_equal(
        manifest_protocol.get("protocol"),
        SOURCE_PROTOCOL,
        "source manifest protocol",
        manifest_path,
    )
    recorded_sources = manifest.get("source_sha256")
    if not isinstance(recorded_sources, Mapping):
        raise ValueError(f"Missing source-code hashes in {manifest_path}")
    for relative in (
        "population_scaling.py",
        "utils/official_prompt_protocol.py",
        "utils/official_randopt_protocol.py",
        "utils/recursive_modular_v2.py",
        "utils/worker_extn.py",
    ):
        expected_digest = recorded_sources.get(relative)
        if not isinstance(expected_digest, str):
            raise ValueError(f"Source manifest does not bind {relative}: {manifest_path}")
        current_digest = _sha256(REPO_ROOT / relative)
        if current_digest != expected_digest:
            raise ValueError(
                f"Inference-critical source changed since candidate search: {relative}; "
                f"actual={current_digest}, expected={expected_digest}"
            )

    prompts = results.get("prompt_tokenization")
    if not isinstance(prompts, Mapping):
        raise ValueError(f"Missing source prompt manifests: {results_path}")
    train_manifest = prompts.get("train")
    test_manifest = prompts.get("validation")
    if not isinstance(train_manifest, Mapping) or not isinstance(test_manifest, Mapping):
        raise ValueError(f"Incomplete source prompt manifests: {results_path}")
    _validate_prompt_manifest(
        train_manifest,
        examples=TRAIN_SAMPLES,
        tokens=int(config["train_prompt_tokens"]),
        digest=str(config["train_prompt_sha256"]),
        label="source train",
        path=results_path,
    )
    _validate_prompt_manifest(
        test_manifest,
        examples=int(config["test_samples"]),
        tokens=int(config["test_prompt_tokens"]),
        digest=str(config["test_prompt_sha256"]),
        label="source validation",
        path=results_path,
    )

    _require_equal(len(candidates), SOURCE_POPULATION_SIZE, "source candidate count", candidate_path)
    expected_seeds = build_candidate_seeds(seed, SOURCE_POPULATION_SIZE)
    for index, (row, expected_seed) in enumerate(zip(candidates, expected_seeds)):
        _require_equal(
            int(row.get("candidate_index", -1)), index, "source candidate index", candidate_path
        )
        _require_equal(int(row.get("seed", -1)), expected_seed, "source candidate seed", candidate_path)
        _require_close(row.get("radius"), RADIUS, "source candidate radius", candidate_path)
        reward = float(row.get("train_reward", float("nan")))
        if not math.isfinite(reward):
            raise ValueError(f"Non-finite source train reward at candidate {index}")

    selections, selected_union = select_prefix_experts(
        candidates, config["prefixes"], max_k=max(TOP_K_VALUES)
    )
    source_top1 = select_prefix_experts(
        candidates, (SOURCE_POPULATION_SIZE,), max_k=1
    )[0][SOURCE_POPULATION_SIZE][0]
    endpoint = (
        results.get("population_results", {})
        .get(str(SOURCE_POPULATION_SIZE), {})
        .get("ensemble_results", {})
        .get("1")
    )
    if not isinstance(endpoint, Mapping):
        raise ValueError(f"Missing source N=3000/K=1 endpoint: {results_path}")
    _require_equal(
        endpoint.get("selected_candidate_indices"),
        [int(source_top1["candidate_index"])],
        "source top-1 candidate",
        results_path,
    )
    _require_equal(
        endpoint.get("selected_seeds"), [int(source_top1["seed"])], "source top-1 seed", results_path
    )
    _require_equal(int(endpoint.get("num_samples", -1)), int(config["test_samples"]), "source endpoint samples", results_path)
    recorded_statistics = (
        results.get("population_results", {})
        .get(str(SOURCE_POPULATION_SIZE), {})
        .get("candidate_statistics")
    )
    if not isinstance(recorded_statistics, Mapping):
        raise ValueError(f"Missing source candidate statistics: {results_path}")
    recomputed_statistics = population_runner.candidate_summary(candidates, None)
    for key, recomputed in recomputed_statistics.items():
        recorded = recorded_statistics.get(key)
        if recomputed is None:
            _require_equal(recorded, None, f"source candidate statistic {key}", results_path)
            continue
        try:
            recorded_float = float(recorded)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source candidate statistic {key}: {results_path}") from exc
        if not math.isclose(
            recorded_float,
            float(recomputed),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Source candidate statistic {key} changed: {results_path}")

    serialized_source = source_selection.get(str(SOURCE_POPULATION_SIZE))
    if not isinstance(serialized_source, list) or len(serialized_source) != 1:
        raise ValueError(f"Invalid source selection artifact: {selection_path}")
    _require_equal(
        int(serialized_source[0].get("candidate_index", -1)),
        int(source_top1["candidate_index"]),
        "source serialized top-1",
        selection_path,
    )

    _require_equal(len(cached_rows), 1, "source cached expert count", expert_path)
    cached = cached_rows[0]
    _require_equal(int(cached.get("seed", -1)), int(source_top1["seed"]), "source cached seed", expert_path)
    _require_equal(
        int(cached.get("candidate_index", -1)),
        int(source_top1["candidate_index"]),
        "source cached candidate",
        expert_path,
    )
    answers = cached.get("answers")
    correctness = cached.get("correct")
    if not isinstance(answers, list) or not isinstance(correctness, list):
        raise ValueError(f"Invalid source cached prediction arrays: {expert_path}")
    _require_equal(len(answers), int(config["test_samples"]), "source cached answers", expert_path)
    _require_equal(len(correctness), int(config["test_samples"]), "source cached correctness", expert_path)
    correct = sum(bool(value) for value in correctness)
    _require_equal(correct, int(endpoint.get("correct", -1)), "source cached correct count", expert_path)
    accuracy = float(endpoint.get("accuracy", float("nan")))
    if not math.isclose(
        accuracy,
        correct / int(config["test_samples"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Source cached prediction disagrees with endpoint accuracy: {expert_path}")

    local_profile = load_sensitivity_profile(SENSITIVITY_PROFILE_PATH)
    loaded_source_profile = load_sensitivity_profile(source_profile)
    if loaded_source_profile != local_profile:
        raise ValueError(f"Source sensitivity profile differs from fixed profile: {profile_path}")
    _require_equal(
        sensitivity_profile_fingerprint(loaded_source_profile),
        SENSITIVITY_PROFILE_FINGERPRINT,
        "source profile fingerprint",
        profile_path,
    )

    source_files = {
        "results.json": results_path,
        "args.json": args_path,
        "matched_run_manifest.json": manifest_path,
        "candidate_records.jsonl": candidate_path,
        "expert_predictions.jsonl": expert_path,
        "selected_models_by_prefix.json": selection_path,
        "sensitivity_profile.json": profile_path,
    }
    return {
        "task": task,
        "seed": seed,
        "run_dir": source_run_dir,
        "results": results,
        "args": source_args,
        "profile": loaded_source_profile,
        "candidates": candidates,
        "selections": selections,
        "selected_union": selected_union,
        "cached_rows": cached_rows,
        "source_sha256": {name: _sha256(path) for name, path in source_files.items()},
    }


def _serialize_selections(
    selections: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    serialized: dict[str, list[dict[str, Any]]] = {}
    for prefix, rows in selections.items():
        serialized[str(prefix)] = [
            {
                "rank": rank,
                "candidate_index": int(row["candidate_index"]),
                "seed": int(row["seed"]),
                "radius": float(row["radius"]),
                "train_reward": float(row["train_reward"]),
            }
            for rank, row in enumerate(rows, 1)
        ]
    return serialized


def _validate_current_prompts(
    source: Mapping[str, Any],
) -> tuple[Any, list[Any], list[Any], Mapping[str, Any], Mapping[str, Any]]:
    task = str(source["task"])
    config = TASK_CONFIG[task]
    handler = get_dataset_handler(task)
    data_args = argparse.Namespace(
        train_data_path=str(config["train_path"]),
        test_data_path=str(config["test_path"]),
        train_samples=TRAIN_SAMPLES,
        test_samples=None,
    )
    train_datas, validation_datas = population_runner.load_data(handler, data_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    train_rendered = population_runner.format_prompts(
        tokenizer, MODEL, train_datas, chat_template_date=CHAT_TEMPLATE_DATE
    )
    validation_rendered = population_runner.format_prompts(
        tokenizer, MODEL, validation_datas, chat_template_date=CHAT_TEMPLATE_DATE
    )
    _, train_token_ids = population_runner.prepare_generation_prompts(tokenizer, train_rendered)
    validation_prompts, validation_token_ids = population_runner.prepare_generation_prompts(
        tokenizer, validation_rendered
    )
    train_prompt_manifest = prompt_manifest(
        tokenizer, train_token_ids, chat_template_date=CHAT_TEMPLATE_DATE
    )
    validation_prompt_manifest = prompt_manifest(
        tokenizer, validation_token_ids, chat_template_date=CHAT_TEMPLATE_DATE
    )
    _validate_prompt_manifest(
        train_prompt_manifest,
        examples=TRAIN_SAMPLES,
        tokens=int(config["train_prompt_tokens"]),
        digest=str(config["train_prompt_sha256"]),
        label="current train",
        path=Path("current prompt construction"),
    )
    _validate_prompt_manifest(
        validation_prompt_manifest,
        examples=int(config["test_samples"]),
        tokens=int(config["test_prompt_tokens"]),
        digest=str(config["test_prompt_sha256"]),
        label="current validation",
        path=Path("current prompt construction"),
    )
    source_prompts = source["results"]["prompt_tokenization"]
    for label, actual, expected in (
        ("train", train_prompt_manifest, source_prompts["train"]),
        ("validation", validation_prompt_manifest, source_prompts["validation"]),
    ):
        for key in (
            "scheme",
            "chat_template_date",
            "num_prompts",
            "total_tokens",
            "token_ids_sha256",
        ):
            _require_equal(actual.get(key), expected.get(key), f"current/source {label} {key}", Path("prompt manifests"))
    return (
        handler,
        validation_datas,
        validation_prompts,
        train_prompt_manifest,
        validation_prompt_manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Completed N=3000/K=1 MN-RandOpt runner directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New derived run directory (must not exist)"
    )
    parser.add_argument("--cuda-devices", default="0", help="Exactly one CUDA device")
    parser.add_argument(
        "--os-wall-time-path",
        type=Path,
        help="GNU time record written by the matrix launcher after process exit",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate source artifacts and print the inference plan without creating files/GPU work",
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if not args.cuda_devices or "," in str(args.cuda_devices):
        raise ValueError("Exactly one CUDA device is required")
    if not args.validate_only:
        if args.output_dir is None:
            raise ValueError("--output-dir is required unless --validate-only is used")
        if args.os_wall_time_path is None:
            raise ValueError("--os-wall-time-path is required for measured runs")
        if args.output_dir.expanduser().resolve().exists():
            raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    total_start = time.perf_counter()
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
    os.environ.setdefault("PYTHONHASHSEED", "0")

    source = validate_source_run(args.source_run_dir)
    task = str(source["task"])
    seed = int(source["seed"])
    config = TASK_CONFIG[task]
    selections = source["selections"]
    selected_union = source["selected_union"]
    cached_by_seed, missing_experts = plan_prediction_cache(
        selected_union, source["cached_rows"]
    )
    budgets = build_fresh_compute_budgets(task)

    print(
        f"Validated source task={task} seed={seed} N=3000/K=1; "
        f"search inference to execute: 0"
    )
    print(
        f"Derived prefixes={list(config['prefixes'])}, K={list(TOP_K_VALUES)}, "
        f"unique experts={len(selected_union)}, cache hits={len(cached_by_seed)}, "
        f"new held-out experts={len(missing_experts)}"
    )
    if args.validate_only:
        print("Validation-only complete; no output files or GPU workers were created.")
        return 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    handler, validation_datas, validation_prompts, train_prompt_manifest, validation_prompt_manifest = (
        _validate_current_prompts(source)
    )
    for cached_seed, cached_row in cached_by_seed.items():
        answers = list(cached_row["answers"])
        recomputed = [
            population_runner.answer_is_correct(handler, answer, data)
            for answer, data in zip(answers, validation_datas)
        ]
        if recomputed != [bool(value) for value in cached_row["correct"]]:
            raise ValueError(
                f"Cached correctness no longer reproduces for selected seed {cached_seed}"
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(output_dir / "selected_models_by_prefix.json", _serialize_selections(selections))
    _write_jsonl(output_dir / "expert_predictions.jsonl", cached_by_seed.values())

    source_validation = {
        "status": "validated",
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source_run_dir": str(source["run_dir"]),
        "source_sha256": source["source_sha256"],
        "task": task,
        "seed": seed,
        "source_candidate_count": len(source["candidates"]),
        "search_inference_executed": False,
        "search_candidate_evaluations_executed": 0,
    }
    _write_json_atomic(output_dir / "source_validation.json", source_validation)
    _write_json_atomic(
        output_dir / "args.json",
        {
            "source_run_dir": str(source["run_dir"]),
            "output_dir": str(output_dir),
            "cuda_devices": str(args.cuda_devices),
            "os_wall_time_path": str(args.os_wall_time_path.expanduser().resolve()),
            "task": task,
            "seed": seed,
            "population_prefixes": list(config["prefixes"]),
            "top_k_values": list(TOP_K_VALUES),
            "reuse_source_predictions": True,
        },
    )
    cache_manifest = {
        "source_prediction_file": str(source["run_dir"] / "expert_predictions.jsonl"),
        "selected_union_count": len(selected_union),
        "cache_hit_count": len(cached_by_seed),
        "new_heldout_expert_count": len(missing_experts),
        "cache_hit_seeds": sorted(cached_by_seed),
        "new_heldout_expert_seeds": [int(row["seed"]) for row in missing_experts],
    }
    _write_json_atomic(output_dir / "expert_cache_manifest.json", cache_manifest)
    _write_json_atomic(
        output_dir / "derived_run_manifest.json",
        {
            "schema_version": "mn-randopt-es-at-scale-ensemble-eval-run-v1",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "source_validation": source_validation,
            "cache": cache_manifest,
        },
    )

    eval_args = argparse.Namespace(
        radius=RADIUS,
        perturbation_method=PERTURBATION_METHOD,
        mass_config=MASS_CONFIG,
        sensitivity_profile=source["profile"],
        power_iterations=POWER_ITERATIONS,
        num_engines=1,
        top_k_list=list(TOP_K_VALUES),
        population_prefix_list=list(config["prefixes"]),
    )
    sampling_params = SamplingParams(temperature=0.0, seed=seed, max_tokens=MAX_TOKENS)
    timings: dict[str, float] = {}
    answers_by_seed = {
        cached_seed: list(row["answers"]) for cached_seed, row in cached_by_seed.items()
    }
    expert_completion_tokens = 0
    population_results: dict[str, Any]
    engines: Sequence[Any] = []
    placement_groups: Sequence[Any] = []
    environment: dict[str, Any] = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "python": sys.version,
        "torch": torch.__version__,
        "ray": importlib.metadata.version("ray"),
        "vllm": importlib.metadata.version("vllm"),
        "transformers": importlib.metadata.version("transformers"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "dtype": PRECISION,
        "inference_backend": "vllm",
        "cuda_visible_devices": str(args.cuda_devices),
        "num_visible_gpus": 1,
        "num_engines": 1,
        "tensor_parallel_size": 1,
        "search_inference_executed": False,
        "source_run_dir": str(source["run_dir"]),
        "source_sha256": source["source_sha256"],
    }

    setup_start = time.perf_counter()
    try:
        if os.environ.get("RAY_ADDRESS"):
            ray.init(address="auto", ignore_reinit_error=True)
        else:
            ray.init(address="local", ignore_reinit_error=True)
        engines, placement_groups = launch_engines(
            1,
            MODEL,
            precision=PRECISION,
            tensor_parallel_size=1,
            revision=MODEL_REVISION,
        )
        population_runner._synchronize_cuda_engines(engines)
        timings["setup_and_engine_launch_sec"] = time.perf_counter() - setup_start
        environment["accelerators"] = population_runner._collect_accelerator_metadata(engines)

        heldout_start = time.perf_counter()
        new_answers, expert_timing, expert_completion_tokens = (
            population_runner.evaluate_selected_experts(
                args=eval_args,
                engines=engines,
                handler=handler,
                validation_prompts=validation_prompts,
                validation_datas=validation_datas,
                sampling_params=sampling_params,
                selected_union=missing_experts,
                run_dir=output_dir,
                save_predictions=True,
            )
        )
        population_runner._synchronize_cuda_engines(engines)
        timings.update(expert_timing)
        timings["incremental_heldout_evaluation_sec"] = time.perf_counter() - heldout_start
        answers_by_seed.update(new_answers)
        missing_answers = sorted(
            int(row["seed"])
            for row in selected_union
            if int(row["seed"]) not in answers_by_seed
        )
        if missing_answers:
            raise RuntimeError(f"Missing held-out predictions after evaluation: {missing_answers}")

        population_results, voting_sec = population_runner.evaluate_prefix_ensembles(
            args=eval_args,
            handler=handler,
            validation_datas=validation_datas,
            selections=selections,
            answers_by_seed=answers_by_seed,
            base_validation_accuracy=None,
            run_dir=output_dir,
            save_predictions=True,
            all_records=source["candidates"],
            base_train_reward=None,
        )
        timings["prefix_voting_sec"] = voting_sec
    finally:
        cleanup_start = time.perf_counter()
        if engines:
            cleanup_engines(list(engines), list(placement_groups))
        elif ray.is_initialized():
            ray.shutdown()
        timings["cleanup_sec"] = time.perf_counter() - cleanup_start

    timings["incremental_total_wall_time_sec"] = time.perf_counter() - total_start
    timings["incremental_gpu_hours"] = timings["incremental_total_wall_time_sec"] / 3600.0
    validation_prompt_tokens = int(config["test_prompt_tokens"])
    token_counts = {
        "search_prompt_tokens_executed": 0,
        "cache_hit_validation_prompt_tokens_not_executed": (
            len(cached_by_seed) * validation_prompt_tokens
        ),
        "incremental_validation_prompt_tokens": (
            len(missing_experts) * validation_prompt_tokens
        ),
        "incremental_validation_completion_tokens": expert_completion_tokens,
        "fresh_union_validation_prompt_tokens": (
            len(selected_union) * validation_prompt_tokens
        ),
    }
    results = {
        "schema_version": "mn-randopt-es-at-scale-ensemble-eval-results-v1",
        "status": "complete",
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "method": "Modular Norm RandOpt",
        "dataset": task,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "global_seed": seed,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": len(validation_datas),
        "population_size": SOURCE_POPULATION_SIZE,
        "population_prefixes": list(config["prefixes"]),
        "top_k_values": list(TOP_K_VALUES),
        "unique_validation_experts": len(selected_union),
        "cached_validation_experts": len(cached_by_seed),
        "new_validation_experts": len(missing_experts),
        "search_reused": True,
        "search_inference_executed": False,
        "search_candidate_evaluations_executed": 0,
        "perturbation_method": PERTURBATION_METHOD,
        "radius": RADIUS,
        "mass_config": MASS_CONFIG,
        "sensitivity_profile_fingerprint": SENSITIVITY_PROFILE_FINGERPRINT,
        "power_iterations": POWER_ITERATIONS,
        "precision": PRECISION,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "chat_template_date": CHAT_TEMPLATE_DATE,
        "prompt_tokenization": {
            "train": train_prompt_manifest,
            "validation": validation_prompt_manifest,
        },
        "fresh_compute_budgets": budgets,
        "fresh_compute_budget_note": (
            "Budget rows are from-scratch model-prompt evaluation counts. "
            "They are distinct from this cache-aware incremental run cost."
        ),
        "population_results": population_results,
        "source_validation": source_validation,
        "cache": cache_manifest,
        "timing": timings,
        "token_counts": token_counts,
        "environment": environment,
        "external_wall_clock": {
            "path": str(args.os_wall_time_path.expanduser().resolve()),
            "record_available_after_process_exit": True,
        },
    }
    _write_json_atomic(output_dir / "timing.json", timings)
    _write_json_atomic(output_dir / "token_counts.json", token_counts)
    _write_json_atomic(output_dir / "environment.json", environment)
    _write_json_atomic(output_dir / "results.json", results)
    _write_json_atomic(
        output_dir / "derived_run_manifest.json",
        {
            "schema_version": "mn-randopt-es-at-scale-ensemble-eval-run-v1",
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "results_path": str((output_dir / "results.json").resolve()),
            "source_validation": source_validation,
            "cache": cache_manifest,
        },
    )
    print(f"Derived ensemble evaluation complete: {output_dir / 'results.json'}")
    print(
        f"Incremental held-out experts: {len(missing_experts)}; "
        f"incremental GPU-hours: {timings['incremental_gpu_hours']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
