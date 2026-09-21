#!/usr/bin/env python3
"""Adapter for the ES-at-Scale-matched MN-RandOpt timing protocol.

``population_scaling.py`` already contains synchronized phase timing, but its
public ``--wall_clock_mode`` intentionally accepts only the separate K=25
population-efficiency experiment.  This adapter installs a stricter N=300,
K=1 validator and protocol name without changing that established experiment.
It then appends matched-protocol provenance after the underlying runner has
cleaned up its GPU workers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import population_scaling as runner  # noqa: E402
from utils.perturbation_norms import sensitivity_profile_fingerprint  # noqa: E402


PROTOCOL = "mn-randopt-es-at-scale-matched-v1"
ES_ITERATIONS = 10
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CHAT_TEMPLATE_DATE = "11 Aug 2026"
PERTURBATION_METHOD = "recursive_modular_shell_v2"
RADIUS = 0.16
POPULATION_SIZE = 300
TOP_K = 1
TRAIN_SAMPLES = 200
MAX_TOKENS = 1024
PRECISION = "bfloat16"
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
ADDITIONAL_SOURCE_PATHS: tuple[Path, ...] = ()

TASK_CONFIG: Mapping[str, Mapping[str, Any]] = {
    "countdown": {
        "train_path": REPO_ROOT / "data/countdown/countdown_train.json",
        "test_path": REPO_ROOT / "data/countdown/countdown_test.json",
        "train_sha256": "86f24778f6b545ae9ddb4c3fce6d3e7ce66fd0786632d5df67ca559c7c16337c",
        "test_sha256": "c2f9a47a63fd077f78ff51fac29955916ad7b23e96e5cb0022294dd86632c8bd",
        "test_samples": 1_500,
    },
    "gsm8k": {
        "train_path": REPO_ROOT / "data/gsm8k/train_200.parquet",
        "test_path": REPO_ROOT / "data/gsm8k/test.parquet",
        "train_sha256": "4aff68b6180f627c84444e3384b7ed6ae1fcf082d8a456aeebd8022afd478205",
        "test_sha256": "09cb3b2cd84ec2c679d600e79c094c80606d31497e78fcdf4bc4ab787c92e91f",
        "test_samples": 1_319,
    },
}


for _config in TASK_CONFIG.values():
    for _key in ("train_path", "test_path"):
        _config[_key] = Path(os.environ.get("MN_RANDOPT_DATA", REPO_ROOT / "data")) / Path(_config[_key]).relative_to(REPO_ROOT / "data")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_matched_wall_clock_configuration(
    *,
    dataset: str,
    model_name: str,
    perturbation_method: str,
    radius: float,
    population_size: int,
    population_prefixes: Sequence[int],
    top_k_values: Sequence[int],
    global_seed: int,
    train_samples: int,
    test_samples: int | None,
    max_tokens: int | None,
    precision: str,
    base_only: bool,
) -> None:
    """Fail closed unless the synchronized runner matches the selected ES budget."""

    expected = {
        "model": (model_name, MODEL),
        "perturbation method": (perturbation_method, PERTURBATION_METHOD),
        "population size": (population_size, POPULATION_SIZE),
        "population prefixes": (list(population_prefixes), [POPULATION_SIZE]),
        "top-K": (list(top_k_values), [TOP_K]),
        "train samples": (train_samples, TRAIN_SAMPLES),
        "test-sample override": (test_samples, None),
        "max tokens": (max_tokens, MAX_TOKENS),
        "precision": (precision, PRECISION),
        "base-only": (base_only, False),
    }
    for label, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(
                f"ES-matched MN-RandOpt requires {label}={wanted!r}; got {actual!r}"
            )
    if dataset not in TASK_CONFIG:
        raise ValueError("ES-matched MN-RandOpt supports only Countdown and GSM8K")
    if global_seed not in SEEDS:
        raise ValueError(f"ES-matched final seed must be in {SEEDS}; got {global_seed}")
    if not math.isclose(float(radius), RADIUS, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"ES-matched MN-RandOpt requires radius={RADIUS}; got {radius}")


def _install_matched_sampling_seed_pool() -> None:
    """Make wall-clock sampling use the full matched population seed pool.

    The established K=25 wall-clock runner deliberately takes a prefix from an
    Nmax=300 pool.  The ES-final adapter instead needs all 3,000 deterministic
    seeds, while preserving the established runner unchanged on disk.
    """

    original_run_sampling = runner.run_sampling

    def run_sampling_with_matched_pool(args: Any, *positional: Any, **keywords: Any):
        original_build_candidate_seeds = runner.build_candidate_seeds

        def build_matched_candidate_seeds(
            global_seed: int, requested_pool_size: int
        ) -> list[int]:
            pool_size = int(requested_pool_size)
            if args.wall_clock_mode and pool_size == 300:
                pool_size = int(args.population_size)
            return original_build_candidate_seeds(global_seed, pool_size)

        runner.build_candidate_seeds = build_matched_candidate_seeds
        try:
            return original_run_sampling(args, *positional, **keywords)
        finally:
            runner.build_candidate_seeds = original_build_candidate_seeds

    runner.run_sampling = run_sampling_with_matched_pool


def _validate_parsed_args(args: Any) -> None:
    config = TASK_CONFIG[args.dataset]
    checks = {
        "model revision": (args.model_revision, MODEL_REVISION),
        "chat-template date": (args.chat_template_date, CHAT_TEMPLATE_DATE),
        "number of engines": (args.num_engines, 1),
        "tensor parallelism": (args.tp, 1),
        "power iterations": (args.power_iterations, 8),
        "mass config": (args.mass_config, MASS_CONFIG),
        "sensitivity profile": (
            sensitivity_profile_fingerprint(args.sensitivity_profile),
            SENSITIVITY_PROFILE_FINGERPRINT,
        ),
        "train path": (
            Path(args.train_data_path).expanduser().resolve(),
            Path(config["train_path"]).resolve(),
        ),
        "test path": (
            Path(args.test_data_path).expanduser().resolve(),
            Path(config["test_path"]).resolve(),
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"ES-matched MN-RandOpt {label} mismatch: "
                f"actual={actual!r}, expected={expected!r}"
            )
    if not args.os_wall_time_path:
        raise ValueError("ES-matched runs require --os_wall_time_path")
    if "," in str(args.cuda_devices) or not str(args.cuda_devices):
        raise ValueError("ES-matched runs require exactly one CUDA device")
    for role in ("train", "test"):
        path = Path(config[f"{role}_path"])
        actual = _sha256(path)
        expected = str(config[f"{role}_sha256"])
        if actual != expected:
            raise ValueError(
                f"Fixed {args.dataset} {role} data changed: "
                f"actual={actual}, expected={expected}"
            )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.matched.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_matched_provenance(args: Any, results_path: Path) -> None:
    run_dir = results_path.parent
    results = dict(json.loads(results_path.read_text(encoding="utf-8")))
    environment_path = run_dir / "environment.json"
    wall_clock_path = run_dir / "wall_clock.json"
    environment = dict(json.loads(environment_path.read_text(encoding="utf-8")))
    wall_clock = dict(json.loads(wall_clock_path.read_text(encoding="utf-8")))

    candidate_records_path = run_dir / "candidate_records.jsonl"
    candidate_records = [
        json.loads(line)
        for line in candidate_records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(candidate_records) != POPULATION_SIZE:
        raise RuntimeError(
            "Matched run produced an incomplete candidate population: "
            f"actual={len(candidate_records)}, expected={POPULATION_SIZE}"
        )
    expected_seeds = runner.build_candidate_seeds(args.global_seed, POPULATION_SIZE)
    actual_seeds = [int(record["seed"]) for record in candidate_records]
    if actual_seeds != expected_seeds:
        raise RuntimeError("Matched run candidate seed sequence is invalid")

    source_paths = (
        Path("population_scaling.py"),
        Path("utils/official_prompt_protocol.py"),
        Path("utils/official_randopt_protocol.py"),
        Path("utils/recursive_modular_v2.py"),
        Path("utils/worker_extn.py"),
        Path("scripts/baselines/es_at_scale/run_mn_randopt_matched.py"),
        Path("scripts/run_es_comparison.py"),
    ) + ADDITIONAL_SOURCE_PATHS
    matched_source_sha256 = {
        str(path): _sha256(REPO_ROOT / path) for path in source_paths
    }
    config = TASK_CONFIG[args.dataset]
    matched_protocol = {
        "protocol": PROTOCOL,
        "comparison_target": f"ES-at-Scale iteration {ES_ITERATIONS}",
        "es_iterations": ES_ITERATIONS,
        "candidate_evaluations": POPULATION_SIZE,
        "model_prompt_evaluations": POPULATION_SIZE * TRAIN_SAMPLES,
        "held_out_evaluation_excluded_from_search_budget": True,
        "single_model_endpoint": True,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "task": args.dataset,
        "seed": args.global_seed,
        "train_samples": TRAIN_SAMPLES,
        "test_samples": int(config["test_samples"]),
        "population_size": POPULATION_SIZE,
        "top_k": TOP_K,
        "radius": RADIUS,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "precision": PRECISION,
        "chat_template_date": CHAT_TEMPLATE_DATE,
        "setup_includes_base_evaluation": False,
        "wall_clock": {
            "internal": "GPU-synchronized phase timing",
            "external": "GNU /usr/bin/time",
            "os_time_path": str(Path(args.os_wall_time_path).resolve()),
        },
    }
    environment["wall_clock_protocol"] = PROTOCOL
    environment["wall_clock_source_sha256"] = matched_source_sha256
    wall_clock["protocol"] = PROTOCOL
    wall_clock["candidate_seed_pool_size"] = POPULATION_SIZE
    wall_clock["matched_protocol"] = matched_protocol
    wall_clock["environment"] = environment
    results["status"] = "complete"
    results["candidate_seed_pool_size"] = POPULATION_SIZE
    results["matched_protocol"] = matched_protocol
    results["environment"] = environment
    results["wall_clock"] = wall_clock

    manifest = {
        "schema_version": "mn-randopt-es-at-scale-matched-run-v1",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "results_path": str(results_path.resolve()),
        "matched_protocol": matched_protocol,
        "prompt_tokenization": results["prompt_tokenization"],
        "data": {
            "train_path": str(Path(config["train_path"]).resolve()),
            "train_sha256": config["train_sha256"],
            "test_path": str(Path(config["test_path"]).resolve()),
            "test_sha256": config["test_sha256"],
        },
        "model": {
            "requested_name": MODEL,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": environment.get("resolved_model_revision"),
        },
        "source_sha256": matched_source_sha256,
        "external_wall_clock": {
            "path": str(Path(args.os_wall_time_path).resolve()),
            "record_available_after_process_exit": True,
        },
    }
    _write_json_atomic(environment_path, environment)
    _write_json_atomic(wall_clock_path, wall_clock)
    _write_json_atomic(results_path, results)
    _write_json_atomic(run_dir / "matched_run_manifest.json", manifest)


def main() -> int:
    # Patch only this process.  The established K=25 protocol and its source
    # files remain unchanged.
    runner.WALL_CLOCK_PROTOCOL = PROTOCOL
    runner.validate_wall_clock_configuration = validate_matched_wall_clock_configuration
    _install_matched_sampling_seed_pool()
    args = runner.parse_args()
    _validate_parsed_args(args)

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    before = set(experiment_dir.rglob("results.json")) if experiment_dir.exists() else set()
    runner.main(args)
    after = set(experiment_dir.rglob("results.json"))
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(
            "Underlying runner did not create exactly one result artifact: "
            f"created={created}"
        )
    _append_matched_provenance(args, created[0])
    print(f"Matched run manifest: {created[0].parent / 'matched_run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
