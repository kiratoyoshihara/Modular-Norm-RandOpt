#!/usr/bin/env python3
"""Validate, summarize, and select hyperparameters from ES-at-Scale runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gradient_free.protocol import (
    CALIBRATION_ITERATIONS,
    CALIBRATION_SEEDS,
    COUNTDOWN_SIGMA_GRID,
    DEFAULT_POPULATION_SIZE,
    OFFICIAL_ES_AT_SCALE_COMMIT,
    PROTOCOL_NAME,
)
from utils.gradient_free.result_schema import (
    validate_iteration_record,
    validate_run_manifest,
    validate_summary,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def discover_runs(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
        validate_run_manifest(manifest)
        validate_summary(summary)
        if manifest["protocol"] != PROTOCOL_NAME:
            raise ValueError(f"Protocol mismatch in {manifest_path}")
        if manifest["upstream"]["commit"] != OFFICIAL_ES_AT_SCALE_COMMIT:
            raise ValueError(f"Upstream commit mismatch in {manifest_path}")
        hyper = manifest["hyperparameters"]
        eval_role = next(role for role in manifest["splits"] if role != "train")
        run_row = {
            "run_dir": str(run_dir.resolve()),
            "status": summary["status"],
            "phase": manifest["phase"],
            "task": manifest["task"],
            "model_name": manifest["model"]["requested_name"],
            "model_revision": manifest["model"]["requested_revision"],
            "seed": int(manifest["seed"]),
            "sigma": float(hyper["sigma"]),
            "alpha": float(hyper["alpha"]),
            "population_size": int(hyper["population_size"]),
            "planned_iterations": int(hyper["iterations"]),
            "completed_iterations": int(summary["completed_iterations"]),
            "eval_split": eval_role,
            "eval_split_sha256": manifest["splits"][eval_role]["sha256"],
            "train_split_sha256": manifest["splits"]["train"]["sha256"],
            "train_prompt_sha256": manifest["prompt_tokenization"]["train"][
                "token_ids_sha256"
            ],
            "eval_prompt_sha256": manifest["prompt_tokenization"][eval_role][
                "token_ids_sha256"
            ],
            "protocol_override": bool(manifest.get("protocol_override", False)),
            "candidate_evaluations": int(
                summary["budget"]["completed_candidate_evaluations"]
            ),
            "model_prompt_evaluations": int(
                summary["budget"]["completed_model_prompt_evaluations"]
            ),
            "search_prompt_tokens": int(summary["budget"]["search_prompt_tokens"]),
            "search_completion_tokens": int(
                summary["budget"]["search_completion_tokens"]
            ),
            "wall_clock_sec": float(summary["wall_clock_sec"]),
            "gpu_hours": float(summary.get("gpu_hours", 0.0)),
        }
        runs.append(run_row)

        previous = None
        seen_iterations: set[int] = set()
        for event in _read_jsonl(run_dir / "evaluation_metrics.jsonl"):
            validate_iteration_record(event, previous)
            iteration = int(event["iteration"])
            if iteration in seen_iterations:
                raise ValueError(
                    f"Duplicate evaluation at iteration {iteration} in {run_dir}"
                )
            seen_iterations.add(iteration)
            previous = event
            evaluations.append(
                {
                    **run_row,
                    "method": "ES-at-Scale",
                    "iteration": iteration,
                    "candidate_evaluations": int(event["candidate_evaluations"]),
                    "model_prompt_evaluations": int(
                        event["model_prompt_evaluations"]
                    ),
                    "num_examples": int(event["num_examples"]),
                    "num_correct": int(event["num_correct"]),
                    "accuracy": float(event["accuracy"]),
                    "accuracy_percent": 100.0 * float(event["accuracy"]),
                    "mean_task_reward": float(event["mean_task_reward"]),
                    "evaluation_prompt_tokens": int(event["prompt_tokens"]),
                    "evaluation_completion_tokens": int(event["completion_tokens"]),
                }
            )
    return runs, evaluations


def aggregate_evaluations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "completed" or row["protocol_override"]:
            continue
        key = (
            row["phase"],
            row["task"],
            row["eval_split"],
            float(row["sigma"]),
            int(row["iteration"]),
            int(row["candidate_evaluations"]),
        )
        groups.setdefault(key, []).append(row)

    aggregates = []
    for key, group in sorted(groups.items()):
        phase, task, eval_split, sigma, iteration, candidate_budget = key
        accuracies = [float(row["accuracy"]) for row in group]
        rewards = [float(row["mean_task_reward"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        aggregates.append(
            {
                "method": "ES-at-Scale",
                "phase": phase,
                "task": task,
                "eval_split": eval_split,
                "sigma": sigma,
                "iteration": iteration,
                "candidate_evaluations": candidate_budget,
                "model_prompt_evaluations": int(group[0]["model_prompt_evaluations"]),
                "num_seeds": len(group),
                "seeds": "/".join(str(seed) for seed in seeds),
                "accuracy_mean": statistics.fmean(accuracies),
                "accuracy_sample_sd": statistics.stdev(accuracies)
                if len(accuracies) > 1
                else 0.0,
                "accuracy_mean_percent": 100.0 * statistics.fmean(accuracies),
                "accuracy_sample_sd_percent": 100.0
                * (statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0),
                "mean_task_reward": statistics.fmean(rewards),
                "eval_split_sha256": group[0]["eval_split_sha256"],
                "train_split_sha256": group[0]["train_split_sha256"],
            }
        )
    return aggregates


def select_countdown_sigma(
    evaluations: Sequence[Mapping[str, Any]], output_dir: Path
) -> Mapping[str, Any]:
    target = [
        row
        for row in evaluations
        if row["status"] == "completed"
        and not row["protocol_override"]
        and row["phase"] == "calibration"
        and row["task"] == "countdown"
        and row["eval_split"] == "validation"
        and int(row["iteration"]) == CALIBRATION_ITERATIONS
        and int(row["candidate_evaluations"])
        == CALIBRATION_ITERATIONS * DEFAULT_POPULATION_SIZE
    ]
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for row in target:
        grouped.setdefault(float(row["sigma"]), []).append(row)
    if set(grouped) != set(COUNTDOWN_SIGMA_GRID):
        raise ValueError(
            f"Selection requires all sigmas {COUNTDOWN_SIGMA_GRID}; got {sorted(grouped)}"
        )

    candidates = []
    for sigma in COUNTDOWN_SIGMA_GRID:
        group = grouped[sigma]
        seeds = sorted(int(row["seed"]) for row in group)
        if seeds != list(CALIBRATION_SEEDS):
            raise ValueError(
                f"Sigma {sigma} requires calibration seeds {CALIBRATION_SEEDS}; got {seeds}"
            )
        if len({row["eval_split_sha256"] for row in group}) != 1:
            raise ValueError(f"Validation split changed across sigma {sigma} runs")
        accuracies = [float(row["accuracy"]) for row in group]
        candidates.append(
            {
                "sigma": sigma,
                "mean_validation_accuracy": statistics.fmean(accuracies),
                "sample_sd_validation_accuracy": statistics.stdev(accuracies),
                "seed_accuracies": {
                    str(row["seed"]): float(row["accuracy"]) for row in group
                },
                "run_dirs": [row["run_dir"] for row in group],
                "validation_sha256": group[0]["eval_split_sha256"],
                "train_sha256": group[0]["train_split_sha256"],
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["mean_validation_accuracy"],
            row["sample_sd_validation_accuracy"],
            abs(row["sigma"] - 0.001),
            row["sigma"],
        ),
    )
    selected = ranked[0]
    model_names = {row["model_name"] for row in target}
    model_revisions = {row["model_revision"] for row in target}
    if len(model_names) != 1 or len(model_revisions) != 1:
        raise ValueError("Calibration runs do not use one fixed model revision")
    artifact = {
        "schema_version": "es-hparam-selection-v1",
        "protocol": PROTOCOL_NAME,
        "method": "es-at-scale",
        "task": "countdown",
        "selection_split": "validation",
        "selection_metric": "mean accuracy across seeds at 300 candidate evaluations",
        "selection_rule": (
            "maximize mean accuracy; then minimize sample SD; then minimize "
            "distance to official default sigma=0.001; then choose smaller sigma"
        ),
        "sigma_grid": list(COUNTDOWN_SIGMA_GRID),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "calibration_iterations": CALIBRATION_ITERATIONS,
        "population_size": DEFAULT_POPULATION_SIZE,
        "candidate_evaluations": CALIBRATION_ITERATIONS
        * DEFAULT_POPULATION_SIZE,
        "selected_sigma": selected["sigma"],
        "selected_alpha": selected["sigma"] / 2.0,
        "upstream_commit": OFFICIAL_ES_AT_SCALE_COMMIT,
        "model": {
            "requested_name": next(iter(model_names)),
            "requested_revision": next(iter(model_revisions)),
        },
        "candidates": candidates,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "es_selected_hyperparameters.json"
    _write_json(path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    runs, evaluations = discover_runs(input_root)
    if not runs:
        raise SystemExit(f"No complete run artifacts found under {input_root}")
    aggregates = aggregate_evaluations(evaluations)
    _write_csv(output_dir / "es_runs.csv", runs)
    _write_csv(output_dir / "es_evaluations.csv", evaluations)
    _write_csv(output_dir / "es_aggregate.csv", aggregates)
    print(f"Validated {len(runs)} runs and {len(evaluations)} evaluations")
    if args.select:
        artifact = select_countdown_sigma(evaluations, output_dir)
        print(f"Selected sigma={artifact['selected_sigma']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
