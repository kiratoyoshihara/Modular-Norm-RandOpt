"""Predeclared protocol for the ES-at-Scale comparison.

The protocol intentionally separates hyperparameter calibration from final
evaluation.  Countdown validation is the only split used to choose ``sigma``;
the selected value is then frozen for fresh Countdown and GSM8K runs that both
start from the same pretrained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


PROTOCOL_NAME = "gradient-free-baselines-v1"
OFFICIAL_ES_AT_SCALE_REPOSITORY = "https://github.com/VsonicV/es-at-scale.git"
OFFICIAL_ES_AT_SCALE_COMMIT = "574a9d134da1ffce2a8bb812019899e5c96b588a"
OFFICIAL_ES_TRAINER_SHA256 = (
    "4f7044600af13b0a73f807aabb7c188610438d31501cc8a053d0f6efae14b42e"
)
OFFICIAL_ES_WORKER_SHA256 = (
    "b5e99e4f050f6882529eae98d3b1cd9cb54894362fefcb2b7b802ce3f75a933a"
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_POPULATION_SIZE = 30
DEFAULT_TRAIN_SAMPLES = 200
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PRECISION = "bfloat16"
DEFAULT_REWARD_SHAPING = "z-scores"
DEFAULT_CHAT_TEMPLATE_DATE = "11 Aug 2026"

COUNTDOWN_SIGMA_GRID = (0.0005, 0.001, 0.002)
CALIBRATION_SEEDS = (39, 40, 41)
FINAL_SEEDS = (42, 43, 44)
CALIBRATION_ITERATIONS = 10
FINAL_ITERATIONS = 100
DEFAULT_CHECKPOINT_ITERATIONS = (1, 2, 5, 10, 30, 100)
PRIMARY_CANDIDATE_BUDGETS = (300, 900, 3000)


@dataclass(frozen=True)
class SplitSpec:
    task: str
    role: str
    relative_path: str
    num_examples: int
    sha256: str

    def resolve(self, repo_root: str | Path) -> Path:
        return Path(repo_root) / self.relative_path


SPLIT_SPECS = {
    ("countdown", "train"): SplitSpec(
        task="countdown",
        role="train",
        relative_path="data/countdown/countdown_train.json",
        num_examples=200,
        sha256="86f24778f6b545ae9ddb4c3fce6d3e7ce66fd0786632d5df67ca559c7c16337c",
    ),
    ("countdown", "validation"): SplitSpec(
        task="countdown",
        role="validation",
        relative_path="data/countdown/countdown_validation.json",
        num_examples=500,
        sha256="29186cd3825cb0beb95540c31b32e4d5e59b56387461f5616040cd4ef8193e66",
    ),
    ("countdown", "test"): SplitSpec(
        task="countdown",
        role="test",
        relative_path="data/countdown/countdown_test.json",
        num_examples=1500,
        sha256="c2f9a47a63fd077f78ff51fac29955916ad7b23e96e5cb0022294dd86632c8bd",
    ),
    ("gsm8k", "train"): SplitSpec(
        task="gsm8k",
        role="train",
        relative_path="data/gsm8k/train_200.parquet",
        num_examples=200,
        sha256="4aff68b6180f627c84444e3384b7ed6ae1fcf082d8a456aeebd8022afd478205",
    ),
    ("gsm8k", "test"): SplitSpec(
        task="gsm8k",
        role="test",
        relative_path="data/gsm8k/test.parquet",
        num_examples=1319,
        sha256="09cb3b2cd84ec2c679d600e79c094c80606d31497e78fcdf4bc4ab787c92e91f",
    ),
}


def get_split_spec(task: str, role: str) -> SplitSpec:
    try:
        return SPLIT_SPECS[(task, role)]
    except KeyError as exc:
        available = ", ".join(f"{t}/{r}" for t, r in sorted(SPLIT_SPECS))
        raise ValueError(
            f"No canonical split for {task}/{role}; available: {available}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_file(repo_root: str | Path, spec: SplitSpec) -> Path:
    path = spec.resolve(repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Required {spec.task}/{spec.role} split is missing: {path}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {spec.sha256}, got {actual}"
        )
    return path


def alpha_for_sigma(sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(sigma) / 2.0


def population_seeds(
    global_seed: int, zero_based_iteration: int, population_size: int
) -> tuple[int, ...]:
    """Reproduce the pinned upstream per-iteration perturbation seeds."""

    if zero_based_iteration < 0:
        raise ValueError("zero_based_iteration must be non-negative")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    rng = np.random.default_rng(seed=int(global_seed) + int(zero_based_iteration))
    values = rng.integers(0, 2**30, size=population_size, dtype=np.int64)
    return tuple(int(value) for value in values)


def zscore_rewards(rewards: Iterable[float]) -> tuple[float, ...]:
    """Match upstream population z-scoring, including its 1e-8 stabilizer."""

    values = np.asarray(tuple(float(value) for value in rewards), dtype=np.float64)
    if values.size == 0:
        raise ValueError("rewards must be non-empty")
    mean = float(np.mean(values))
    std = float(np.std(values))
    return tuple(float((value - mean) / (std + 1e-8)) for value in values)


def es_update_scales(rewards: Iterable[float], alpha: float) -> tuple[float, ...]:
    """Return each noise direction's scalar in the official one-sided update."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    normalized = zscore_rewards(rewards)
    population_size = len(normalized)
    return tuple(float(alpha) * value / population_size for value in normalized)


def candidate_evaluations(population_size: int, iterations: int) -> int:
    if population_size <= 0 or iterations < 0:
        raise ValueError("population_size must be positive and iterations non-negative")
    return int(population_size) * int(iterations)


def model_prompt_evaluations(
    population_size: int,
    iterations: int,
    prompts_per_candidate: int,
) -> int:
    if prompts_per_candidate <= 0:
        raise ValueError("prompts_per_candidate must be positive")
    return candidate_evaluations(population_size, iterations) * int(
        prompts_per_candidate
    )


def iterations_for_candidate_budget(budget: int, population_size: int) -> int:
    if budget <= 0 or population_size <= 0:
        raise ValueError("budget and population_size must be positive")
    quotient, remainder = divmod(int(budget), int(population_size))
    if remainder:
        raise ValueError(
            f"Candidate budget {budget} is not divisible by population {population_size}"
        )
    return quotient


def normalize_checkpoint_iterations(
    values: Iterable[int], *, total_iterations: int
) -> tuple[int, ...]:
    if total_iterations < 0:
        raise ValueError("total_iterations must be non-negative")
    normalized = tuple(sorted({int(value) for value in values}))
    if any(value <= 0 for value in normalized):
        raise ValueError("checkpoint iterations must be positive")
    return tuple(value for value in normalized if value <= total_iterations)


def validate_phase_assignment(task: str, phase: str, seed: int) -> str:
    """Return the only evaluation role permitted by the predeclared phase."""

    if phase == "calibration":
        if task != "countdown":
            raise ValueError("Hyperparameter calibration is permitted only on Countdown")
        if seed not in CALIBRATION_SEEDS:
            raise ValueError(
                f"Calibration seed must be one of {CALIBRATION_SEEDS}, got {seed}"
            )
        return "validation"
    if phase == "final":
        if task not in {"countdown", "gsm8k"}:
            raise ValueError(f"Unsupported final task: {task}")
        if seed not in FINAL_SEEDS:
            raise ValueError(f"Final seed must be one of {FINAL_SEEDS}, got {seed}")
        return "test"
    if phase == "smoke":
        return "validation" if task == "countdown" else "test"
    raise ValueError(f"Unknown phase: {phase}")


__all__ = [
    "CALIBRATION_ITERATIONS",
    "CALIBRATION_SEEDS",
    "COUNTDOWN_SIGMA_GRID",
    "DEFAULT_CHAT_TEMPLATE_DATE",
    "DEFAULT_CHECKPOINT_ITERATIONS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_PRECISION",
    "DEFAULT_REWARD_SHAPING",
    "DEFAULT_TRAIN_SAMPLES",
    "FINAL_ITERATIONS",
    "FINAL_SEEDS",
    "OFFICIAL_ES_AT_SCALE_COMMIT",
    "OFFICIAL_ES_AT_SCALE_REPOSITORY",
    "OFFICIAL_ES_TRAINER_SHA256",
    "OFFICIAL_ES_WORKER_SHA256",
    "PRIMARY_CANDIDATE_BUDGETS",
    "PROTOCOL_NAME",
    "SPLIT_SPECS",
    "SplitSpec",
    "alpha_for_sigma",
    "candidate_evaluations",
    "get_split_spec",
    "iterations_for_candidate_budget",
    "model_prompt_evaluations",
    "normalize_checkpoint_iterations",
    "population_seeds",
    "sha256_file",
    "validate_phase_assignment",
    "validate_split_file",
    "es_update_scales",
    "zscore_rewards",
]
