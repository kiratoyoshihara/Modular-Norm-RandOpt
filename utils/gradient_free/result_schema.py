"""Small fail-closed schema checks for ES result artifacts."""

from __future__ import annotations

from typing import Any, Mapping


RESULT_SCHEMA_VERSION = "es-at-scale-result-v1"


def _require(payload: Mapping[str, Any], keys: set[str], *, name: str) -> None:
    missing = sorted(keys.difference(payload))
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def validate_run_manifest(payload: Mapping[str, Any]) -> None:
    _require(
        payload,
        {
            "schema_version",
            "protocol",
            "method",
            "phase",
            "task",
            "model",
            "seed",
            "hyperparameters",
            "splits",
            "prompt_tokenization",
            "upstream",
            "environment",
        },
        name="run manifest",
    )
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {payload['schema_version']}")
    if payload["method"] != "es-at-scale":
        raise ValueError("method must be es-at-scale")
    split_roles = set(payload["splits"])
    if "train" not in split_roles or not ({"validation", "test"} & split_roles):
        raise ValueError("manifest must contain train and one held-out split")


def validate_iteration_record(
    payload: Mapping[str, Any], previous: Mapping[str, Any] | None = None
) -> None:
    _require(
        payload,
        {
            "schema_version",
            "event",
            "iteration",
            "candidate_evaluations",
            "model_prompt_evaluations",
            "elapsed_sec",
            "budget",
        },
        name="iteration record",
    )
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("iteration record schema version mismatch")
    if int(payload["iteration"]) < 0:
        raise ValueError("iteration must be non-negative")
    if int(payload["candidate_evaluations"]) < 0:
        raise ValueError("candidate evaluations must be non-negative")
    if int(payload["model_prompt_evaluations"]) < 0:
        raise ValueError("model–prompt evaluations must be non-negative")
    if previous is not None:
        for key in ("iteration", "candidate_evaluations", "model_prompt_evaluations"):
            if int(payload[key]) < int(previous[key]):
                raise ValueError(f"{key} must be monotonic")


def validate_summary(payload: Mapping[str, Any]) -> None:
    _require(
        payload,
        {
            "schema_version",
            "status",
            "task",
            "phase",
            "seed",
            "completed_iterations",
            "budget",
            "evaluations",
            "wall_clock_sec",
        },
        name="run summary",
    )
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("summary schema version mismatch")
    if payload["status"] not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"Invalid summary status: {payload['status']}")


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "validate_iteration_record",
    "validate_run_manifest",
    "validate_summary",
]
