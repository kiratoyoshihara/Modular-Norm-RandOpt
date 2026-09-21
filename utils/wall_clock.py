"""Canonical K=25 wall-clock protocol and aggregation helpers.

The measurement protocol intentionally runs each population independently:
GSM8K compares Modular Norm RandOpt N=25 with RandOpt N=300, while Countdown
compares N=100 with N=300.  This module is dependency-light so protocol and
reporting tests do not need to import the GPU experiment runner.
"""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


WALL_CLOCK_PROTOCOL = "mn-randopt-wall-clock-k25-v1"
CANONICAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CANONICAL_SEEDS = (42, 43, 44)
CANONICAL_TOP_K = 25

PERTURBATION_METHOD_TO_METHOD = {
    "isotropic": "randopt",
    "recursive_modular_shell_v2": "modular_norm_randopt",
}

CANONICAL_POPULATION_SIZES = {
    "gsm8k": {
        "randopt": 300,
        "modular_norm_randopt": 25,
    },
    "countdown": {
        "randopt": 300,
        "modular_norm_randopt": 100,
    },
}

CANONICAL_RADII = {
    "randopt": 0.0005,
    "modular_norm_randopt": 0.16,
}


def wall_clock_method(perturbation_method: str) -> str:
    """Return the reporting name for a supported perturbation method."""

    try:
        return PERTURBATION_METHOD_TO_METHOD[perturbation_method]
    except KeyError as exc:
        raise ValueError(
            "Wall-clock measurement supports only isotropic RandOpt and "
            "recursive_modular_shell_v2"
        ) from exc


def validate_wall_clock_configuration(
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
    """Fail closed when a timed run differs from the canonical protocol."""

    if base_only:
        raise ValueError("--wall_clock_mode cannot be combined with --base_only")
    if dataset not in CANONICAL_POPULATION_SIZES:
        raise ValueError("Wall-clock mode supports only GSM8K and Countdown")
    if model_name != CANONICAL_MODEL:
        raise ValueError(
            f"Wall-clock mode requires model {CANONICAL_MODEL!r}, got {model_name!r}"
        )

    method = wall_clock_method(perturbation_method)
    expected_population = CANONICAL_POPULATION_SIZES[dataset][method]
    if population_size != expected_population:
        raise ValueError(
            f"Canonical {dataset}/{method} wall-clock population is "
            f"N={expected_population}, got N={population_size}"
        )
    if list(population_prefixes) != [population_size]:
        raise ValueError(
            "Timed runs must use exactly one population prefix equal to "
            "--population_size; generating a larger population and timing a prefix "
            "is forbidden"
        )
    if list(top_k_values) != [CANONICAL_TOP_K]:
        raise ValueError(
            f"Wall-clock mode requires only K={CANONICAL_TOP_K}, got "
            f"{list(top_k_values)}"
        )
    if global_seed not in CANONICAL_SEEDS:
        raise ValueError(
            f"Wall-clock mode requires a seed in {CANONICAL_SEEDS}, got {global_seed}"
        )
    if train_samples != 200:
        raise ValueError("Wall-clock mode requires 200 selection examples")
    if test_samples is not None:
        raise ValueError("Wall-clock mode requires the complete test split")
    if max_tokens not in (None, 1024):
        raise ValueError("Wall-clock mode requires max_tokens=1024")
    if precision != "bfloat16":
        raise ValueError("Wall-clock mode requires bfloat16 precision")

    expected_radius = CANONICAL_RADII[method]
    if not math.isclose(
        float(radius), expected_radius, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            f"Canonical {method} radius is {expected_radius}, got {radius}"
        )


_REQUIRED_RECORD_FIELDS = (
    "protocol",
    "method",
    "perturbation_method",
    "radius",
    "model",
    "task",
    "seed",
    "population_size",
    "candidate_seed_pool_size",
    "evaluated_candidate_count",
    "top_k",
    "train_samples",
    "validation_samples",
    "max_tokens",
    "temperature",
    "chat_template_date",
    "precision",
    "setup_sec",
    "search_sec",
    "ensemble_sec",
    "total_sec",
    "ensemble_accuracy",
    "gpu_synchronized",
    "environment",
)


def validate_wall_clock_record(record: Mapping[str, Any]) -> None:
    """Validate one completed per-run wall-clock record."""

    missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Wall-clock record is missing fields: {missing}")
    if record["protocol"] != WALL_CLOCK_PROTOCOL:
        raise ValueError(f"Unexpected wall-clock protocol: {record['protocol']!r}")
    if record["gpu_synchronized"] is not True:
        raise ValueError("Wall-clock record must confirm GPU synchronization")
    if record["model"] != CANONICAL_MODEL:
        raise ValueError(f"Unexpected wall-clock model: {record['model']!r}")

    task = str(record["task"])
    method = str(record["method"])
    if task not in CANONICAL_POPULATION_SIZES:
        raise ValueError(f"Unexpected wall-clock task: {task!r}")
    if method not in CANONICAL_POPULATION_SIZES[task]:
        raise ValueError(f"Unexpected wall-clock method: {method!r}")
    expected_perturbation_method = {
        value: key for key, value in PERTURBATION_METHOD_TO_METHOD.items()
    }[method]
    if record["perturbation_method"] != expected_perturbation_method:
        raise ValueError(
            "perturbation_method does not match the reported wall-clock method"
        )
    if not math.isclose(
        float(record["radius"]),
        CANONICAL_RADII[method],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Unexpected radius for {method}: {record['radius']}")
    expected_population = CANONICAL_POPULATION_SIZES[task][method]
    if int(record["population_size"]) != expected_population:
        raise ValueError(
            f"Unexpected population for {task}/{method}: "
            f"{record['population_size']} != {expected_population}"
        )
    if int(record["candidate_seed_pool_size"]) != 300:
        raise ValueError("Wall-clock candidate identities must come from the Nmax=300 pool")
    if int(record["evaluated_candidate_count"]) != expected_population:
        raise ValueError(
            "evaluated_candidate_count must equal the independent timed population"
        )
    if int(record["top_k"]) != CANONICAL_TOP_K:
        raise ValueError(f"Unexpected top_k: {record['top_k']}")
    if int(record["seed"]) not in CANONICAL_SEEDS:
        raise ValueError(f"Unexpected seed: {record['seed']}")
    if int(record["train_samples"]) != 200:
        raise ValueError("Wall-clock record must contain 200 selection examples")
    if int(record["validation_samples"]) <= 0:
        raise ValueError("Wall-clock record has no validation examples")
    if int(record["max_tokens"]) != 1024:
        raise ValueError("Wall-clock record must use max_tokens=1024")
    if float(record["temperature"]) != 0.0:
        raise ValueError("Wall-clock record must use greedy decoding")
    if record["precision"] != "bfloat16":
        raise ValueError("Wall-clock record must use bfloat16")

    environment = record["environment"]
    if not isinstance(environment, Mapping):
        raise ValueError("Wall-clock environment must be a mapping")
    if environment.get("inference_backend") != "vllm":
        raise ValueError("Wall-clock inference backend must be vLLM")
    if environment.get("dtype") != "bfloat16":
        raise ValueError("Wall-clock environment dtype must be bfloat16")
    gpu_names = environment.get("gpu_names")
    if not isinstance(gpu_names, list) or not gpu_names:
        raise ValueError("Wall-clock environment must contain GPU names")
    num_visible_gpus = int(environment.get("num_visible_gpus", 0))
    num_engines = int(environment.get("num_engines", 0))
    tensor_parallel_size = int(environment.get("tensor_parallel_size", 0))
    if num_visible_gpus <= 0 or num_engines * tensor_parallel_size != num_visible_gpus:
        raise ValueError("Wall-clock environment has an inconsistent GPU layout")

    phase_values = []
    for field in ("setup_sec", "search_sec", "ensemble_sec", "total_sec"):
        value = float(record[field])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be finite and positive")
        phase_values.append(value)
    accuracy = float(record["ensemble_accuracy"])
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("ensemble_accuracy must be in [0, 1]")

    phase_sum = sum(phase_values[:3])
    total = phase_values[3]
    tolerance = max(1e-6, total * 1e-9)
    if not math.isclose(phase_sum, total, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"total_sec must equal setup+search+ensemble: {total} != {phase_sum}"
        )


def _mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot aggregate an empty sequence")
    return mean(numbers), stdev(numbers) if len(numbers) > 1 else 0.0


def _validate_paired_conditions(
    randopt: Mapping[str, Any],
    modular: Mapping[str, Any],
) -> None:
    """Ensure a speedup pair differs only in method-specific settings."""

    record_fields = (
        "model",
        "model_revision",
        "resolved_model_revision",
        "task",
        "seed",
        "top_k",
        "train_samples",
        "validation_samples",
        "max_tokens",
        "temperature",
        "chat_template_date",
        "precision",
        "candidate_seed_pool_size",
    )
    for field in record_fields:
        if randopt.get(field) != modular.get(field):
            raise ValueError(
                f"Paired wall-clock condition mismatch for {field}: "
                f"{randopt.get(field)!r} != {modular.get(field)!r}"
            )

    randopt_environment = randopt["environment"]
    modular_environment = modular["environment"]
    environment_fields = (
        "torch",
        "vllm",
        "transformers",
        "torch_cuda_version",
        "cuda_visible_devices",
        "num_visible_gpus",
        "num_engines",
        "tensor_parallel_size",
        "dtype",
        "inference_backend",
        "train_data_sha256",
        "test_data_sha256",
        "wall_clock_source_sha256",
    )
    for field in environment_fields:
        if randopt_environment.get(field) != modular_environment.get(field):
            raise ValueError(
                f"Paired environment mismatch for {field}: "
                f"{randopt_environment.get(field)!r} != "
                f"{modular_environment.get(field)!r}"
            )
    if sorted(randopt_environment["gpu_names"]) != sorted(
        modular_environment["gpu_names"]
    ):
        raise ValueError("Paired runs used different GPU models")


def summarize_wall_clock_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate raw times and paired speedups using sample standard deviation."""

    if not records:
        raise ValueError("No wall-clock records were supplied")

    normalized: List[Dict[str, Any]] = []
    keys = set()
    for raw_record in records:
        validate_wall_clock_record(raw_record)
        record = dict(raw_record)
        key = (str(record["task"]), str(record["method"]), int(record["seed"]))
        if key in keys:
            raise ValueError(f"Duplicate wall-clock record for {key}")
        keys.add(key)
        normalized.append(record)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_pair: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in normalized:
        task = str(record["task"])
        method = str(record["method"])
        seed = int(record["seed"])
        grouped[(task, method)].append(record)
        by_pair[(task, seed)][method] = record

    method_summary: List[Dict[str, Any]] = []
    metrics = ("ensemble_accuracy", "setup_sec", "search_sec", "ensemble_sec", "total_sec")
    for (task, method), rows in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "task": task,
            "method": method,
            "population_size": int(rows[0]["population_size"]),
            "top_k": int(rows[0]["top_k"]),
            "num_seeds": len(rows),
        }
        for metric in metrics:
            metric_mean, metric_sd = _mean_sd(float(row[metric]) for row in rows)
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_sd"] = metric_sd
        os_values = [
            float(row["os_elapsed_sec"])
            for row in rows
            if row.get("os_elapsed_sec") is not None
        ]
        if len(os_values) == len(rows):
            os_mean, os_sd = _mean_sd(os_values)
            summary["os_elapsed_sec_mean"] = os_mean
            summary["os_elapsed_sec_sd"] = os_sd
        else:
            summary["os_elapsed_sec_mean"] = None
            summary["os_elapsed_sec_sd"] = None
        method_summary.append(summary)

    paired_rows: List[Dict[str, Any]] = []
    for (task, seed), pair in sorted(by_pair.items()):
        missing = {
            "randopt",
            "modular_norm_randopt",
        } - set(pair)
        if missing:
            raise ValueError(
                f"Missing paired methods for task={task}, seed={seed}: {sorted(missing)}"
            )
        randopt = pair["randopt"]
        modular = pair["modular_norm_randopt"]
        _validate_paired_conditions(randopt, modular)
        paired_rows.append(
            {
                "task": task,
                "seed": seed,
                "search_speedup": float(randopt["search_sec"])
                / float(modular["search_sec"]),
                "total_speedup": float(randopt["total_sec"])
                / float(modular["total_sec"]),
            }
        )

    speedup_summary: List[Dict[str, Any]] = []
    speedups_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        speedups_by_task[str(row["task"])].append(row)
    for task, rows in sorted(speedups_by_task.items()):
        search_mean, search_sd = _mean_sd(row["search_speedup"] for row in rows)
        total_mean, total_sd = _mean_sd(row["total_speedup"] for row in rows)
        speedup_summary.append(
            {
                "task": task,
                "num_pairs": len(rows),
                "search_speedup_mean": search_mean,
                "search_speedup_sd": search_sd,
                "total_speedup_mean": total_mean,
                "total_speedup_sd": total_sd,
            }
        )

    return {
        "protocol": WALL_CLOCK_PROTOCOL,
        "num_runs": len(normalized),
        "method_summary": method_summary,
        "paired_speedups": paired_rows,
        "speedup_summary": speedup_summary,
    }


__all__ = [
    "CANONICAL_MODEL",
    "CANONICAL_POPULATION_SIZES",
    "CANONICAL_RADII",
    "CANONICAL_SEEDS",
    "CANONICAL_TOP_K",
    "PERTURBATION_METHOD_TO_METHOD",
    "WALL_CLOCK_PROTOCOL",
    "summarize_wall_clock_records",
    "validate_wall_clock_configuration",
    "validate_wall_clock_record",
    "wall_clock_method",
]
