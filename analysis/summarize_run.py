#!/usr/bin/env python3
"""Compute paper metrics from the saved ensemble answers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metric(task, answers, targets, handler):
    if len(answers) != len(targets) or not answers:
        raise ValueError("Answers and targets must be nonempty and aligned")
    if task == "rocstories":
        return "exact_match", mean(answer == "".join(target["gold_labels"]) for answer, target in zip(answers, targets))
    if task == "uspto50k":
        recalls = defaultdict(list)
        for answer, target in zip(answers, targets):
            recalls[str(target)].append(str(answer) == str(target))
        return "balanced_accuracy", mean(mean(values) for values in recalls.values())
    if hasattr(handler, "is_voted_answer_correct"):
        scores = [bool(handler.is_voted_answer_correct(a, t)) for a, t in zip(answers, targets)]
    else:
        scores = [bool(handler.is_answer_correct(handler.format_answer_for_check(a), t)) if a else False
                  for a, t in zip(answers, targets)]
    return "accuracy", mean(scores)


def summarize(directory):
    from data_handlers import get_dataset_handler
    args = json.loads((directory / "args.json").read_text())
    handler = get_dataset_handler(args["dataset"])
    train_path, test_path = args["train_data_path"], args["test_data_path"]
    rows = handler.load_data(test_path, split="test")
    if Path(train_path).resolve() == Path(test_path).resolve():
        rows = rows[args["train_samples"]:]
    if args.get("test_samples"):
        rows = rows[:args["test_samples"]]
    groups = defaultdict(list)
    for line in (directory / "ensemble_predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        groups[row["population_size"], row["k"]].append(row)
    results = []
    for (n, k), predictions in sorted(groups.items()):
        predictions.sort(key=lambda row: row["sample_index"])
        if [r["sample_index"] for r in predictions] != list(range(len(rows))):
            raise ValueError("Incomplete or reordered prediction rows")
        name, value = metric(args["dataset"], [r["voted_answer"] for r in predictions],
                             [r["ground_truth"] for r in rows], handler)
        results.append(dict(task=args["dataset"], seed=args["global_seed"], population_size=n, k=k,
                            metric=name, accuracy=value, evaluation_examples=len(rows)))
    (directory / "paper_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Directory containing args.json and ensemble_predictions.jsonl")
    args = p.parse_args()
    print(json.dumps(summarize(args.run), indent=2))


if __name__ == "__main__":
    main()
