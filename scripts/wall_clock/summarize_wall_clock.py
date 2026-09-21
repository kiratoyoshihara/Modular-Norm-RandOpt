#!/usr/bin/env python3
"""Validate and summarize canonical K=25 wall-clock runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wall_clock import (  # noqa: E402
    CANONICAL_SEEDS,
    WALL_CLOCK_PROTOCOL,
    summarize_wall_clock_records,
)


METHODS = ("randopt", "modular_norm_randopt")
METHOD_LABELS = {
    "randopt": "RandOpt",
    "modular_norm_randopt": "Modular Norm RandOpt",
}


def _csv_values(value: str, *, name: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate synchronized GSM8K/Countdown K=25 wall-clock runs"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-tasks",
        default="gsm8k,countdown",
        help="Comma-separated task names expected in the input",
    )
    parser.add_argument(
        "--expected-seeds",
        default=",".join(str(seed) for seed in CANONICAL_SEEDS),
        help="Comma-separated paired seeds expected in the input",
    )
    parser.add_argument(
        "--allow-missing-os-time",
        action="store_true",
        help="Permit records not launched through /usr/bin/time",
    )
    args = parser.parse_args()
    args.expected_task_list = _csv_values(
        args.expected_tasks, name="expected-tasks"
    )
    try:
        args.expected_seed_list = [
            int(value)
            for value in _csv_values(args.expected_seeds, name="expected-seeds")
        ]
    except ValueError as exc:
        parser.error("--expected-seeds must contain integers")
    return args


def _load_os_elapsed(
    record: Mapping[str, Any],
    *,
    allow_missing: bool,
) -> float | None:
    raw_path = record.get("os_wall_time_path")
    if not raw_path:
        if allow_missing:
            return None
        raise ValueError("wall_clock.json does not name an OS timing file")
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Missing /usr/bin/time output: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("exit_status", -1)) != 0:
        raise ValueError(f"OS timing reports a failed command in {path}: {payload}")
    elapsed = float(payload["elapsed_sec"])
    if elapsed < 0.0:
        raise ValueError(f"Negative OS elapsed time in {path}")
    return elapsed


def load_records(input_dir: Path, *, allow_missing_os_time: bool) -> List[Dict[str, Any]]:
    paths = sorted(input_dir.rglob("wall_clock.json"))
    if not paths:
        raise FileNotFoundError(f"No wall_clock.json files found below {input_dir}")
    records: List[Dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        payload["source_path"] = str(path.resolve())
        payload["os_elapsed_sec"] = _load_os_elapsed(
            payload,
            allow_missing=allow_missing_os_time,
        )
        records.append(payload)
    return records


def validate_completeness(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: Sequence[str],
    expected_seeds: Sequence[int],
) -> None:
    expected = {
        (task, method, int(seed))
        for task in expected_tasks
        for method in METHODS
        for seed in expected_seeds
    }
    actual = {
        (str(row["task"]), str(row["method"]), int(row["seed"]))
        for row in records
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or len(records) != len(actual):
        raise ValueError(
            "Wall-clock run set is incomplete or duplicated: "
            f"missing={missing}, unexpected={unexpected}, "
            f"records={len(records)}, unique={len(actual)}"
        )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_mean_sd(row: Mapping[str, Any], metric: str, digits: int = 2) -> str:
    return (
        f"{float(row[f'{metric}_mean']):.{digits}f} ± "
        f"{float(row[f'{metric}_sd']):.{digits}f}"
    )


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# K=25 wall-clock summary",
        "",
        f"Protocol: `{WALL_CLOCK_PROTOCOL}`. Times are seconds; SD is the sample SD across paired seeds.",
        "",
        "| Task | Method | N | K | Accuracy (%) | Search time | Total time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    method_order = {method: index for index, method in enumerate(METHODS)}
    rows = sorted(
        summary["method_summary"],
        key=lambda row: (str(row["task"]), method_order[str(row["method"])]),
    )
    for row in rows:
        accuracy = (
            f"{100.0 * float(row['ensemble_accuracy_mean']):.2f} ± "
            f"{100.0 * float(row['ensemble_accuracy_sd']):.2f}"
        )
        lines.append(
            "| {task} | {method} | {population} | {top_k} | {accuracy} | "
            "{search} | {total} |".format(
                task=row["task"],
                method=METHOD_LABELS[str(row["method"])],
                population=row["population_size"],
                top_k=row["top_k"],
                accuracy=accuracy,
                search=_fmt_mean_sd(row, "search_sec"),
                total=_fmt_mean_sd(row, "total_sec"),
            )
        )

    lines.extend(
        [
            "",
            "| Task | Search speedup (RandOpt / Modular) | Total speedup (RandOpt / Modular) |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["speedup_summary"]:
        lines.append(
            "| {task} | {search:.2f} ± {search_sd:.2f}× | "
            "{total:.2f} ± {total_sd:.2f}× |".format(
                task=row["task"],
                search=float(row["search_speedup_mean"]),
                search_sd=float(row["search_speedup_sd"]),
                total=float(row["total_speedup_mean"]),
                total_sd=float(row["total_speedup_sd"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = load_records(
        args.input_dir,
        allow_missing_os_time=args.allow_missing_os_time,
    )
    validate_completeness(
        records,
        expected_tasks=args.expected_task_list,
        expected_seeds=args.expected_seed_list,
    )
    summary = summarize_wall_clock_records(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wall_clock_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "wall_clock_summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )

    _write_csv(
        args.output_dir / "wall_clock_runs.csv",
        records,
        (
            "task",
            "method",
            "model",
            "seed",
            "population_size",
            "top_k",
            "setup_sec",
            "search_sec",
            "ensemble_sec",
            "total_sec",
            "os_elapsed_sec",
            "ensemble_accuracy",
            "source_path",
            "os_wall_time_path",
        ),
    )
    _write_csv(
        args.output_dir / "wall_clock_method_summary.csv",
        summary["method_summary"],
        (
            "task",
            "method",
            "population_size",
            "top_k",
            "num_seeds",
            "ensemble_accuracy_mean",
            "ensemble_accuracy_sd",
            "setup_sec_mean",
            "setup_sec_sd",
            "search_sec_mean",
            "search_sec_sd",
            "ensemble_sec_mean",
            "ensemble_sec_sd",
            "total_sec_mean",
            "total_sec_sd",
            "os_elapsed_sec_mean",
            "os_elapsed_sec_sd",
        ),
    )
    _write_csv(
        args.output_dir / "wall_clock_paired_speedups.csv",
        summary["paired_speedups"],
        ("task", "seed", "search_speedup", "total_speedup"),
    )
    _write_csv(
        args.output_dir / "wall_clock_speedup_summary.csv",
        summary["speedup_summary"],
        (
            "task",
            "num_pairs",
            "search_speedup_mean",
            "search_speedup_sd",
            "total_speedup_mean",
            "total_speedup_sd",
        ),
    )
    print(f"Validated {len(records)} runs")
    print(f"Summary: {args.output_dir / 'wall_clock_summary.md'}")


if __name__ == "__main__":
    main()
