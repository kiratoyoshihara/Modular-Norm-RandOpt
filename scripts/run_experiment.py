#!/usr/bin/env python3
"""Run one population experiment from the published configuration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mn", "randopt", "attention-only", "mlp-only", "rmsnorm-weights-only",
           "rmsnorm-scale-correction", "frobenius")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments.json")
    p.add_argument("--model", default="qwen-1.5b")
    p.add_argument("--task", default="countdown")
    p.add_argument("--method", choices=METHODS, default="mn")
    p.add_argument("--suite", choices=("transfer", "scaling", "ablation", "smoke"), default="transfer")
    p.add_argument("--seed", type=int, default=42, help="Run one seed; repeat for 42, 43, 44.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="Print the command without loading data or a GPU.")
    return p


def build_command(args):
    cfg = json.loads(args.config.read_text())
    if args.model not in cfg["models"] or args.task not in cfg["tasks"]:
        raise ValueError(f"Choose model from {list(cfg['models'])} and task from {list(cfg['tasks'])}")
    if args.gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    model, task, suite = cfg["models"][args.model], cfg["tasks"][args.task], cfg["suites"][args.suite]
    control = args.method not in ("mn", "randopt")
    if control and not args.model.startswith("qwen-"):
        raise ValueError("These ablation configurations are defined for Qwen models")
    entry = "population_scaling_ablation.py" if control else "population_scaling.py"
    method = "isotropic" if args.method == "randopt" else "recursive_modular_shell_v2"
    radius = 0.0005 if args.method == "randopt" else model["radius"]
    mass = dict(cfg["mass"])
    if args.method == "rmsnorm-weights-only":
        mass = {key: value if key == "norm" else 0.0 for key, value in mass.items()}
    if args.method == "rmsnorm-scale-correction":
        method = "recursive_modular_shell_rmsnorm_only"
    if args.method == "frobenius":
        method, radius = "frobenius", 0.5
    options = dict(dataset=args.task, model_name=model["name"], model_revision=model["revision"],
        train_data_path=str(args.data_root.resolve() / task["train"]),
        test_data_path=str(args.data_root.resolve() / task["test"]),
        precision=cfg["precision"], train_samples=suite.get("train_samples", cfg["train_samples"]),
        max_tokens=suite.get("max_tokens", task["max_tokens"]),
        population_size=suite["population"], population_prefixes=",".join(map(str, suite["prefixes"])),
        top_k_values=",".join(map(str, suite["ensemble_sizes"])), perturbation_method=method,
        radius=radius, power_iterations=cfg["power_iterations"], mass_config=json.dumps(mass),
        num_engines=1, tp=1, cuda_devices=args.gpu, global_seed=args.seed,
        chat_template_date=cfg["chat_template_date"], experiment_dir=str(args.output.resolve()))
    if method == "recursive_modular_shell_v2":
        options["sensitivity_profile"] = str(ROOT / model["profile"])
    if args.method in ("attention-only", "mlp-only"):
        options["parameter_mask"] = args.method.removesuffix("-only")
    if "test_samples" in suite:
        options["test_samples"] = suite["test_samples"]
    return [sys.executable, str(ROOT / entry),
            *(item for key, value in options.items() for item in (f"--{key}", str(value)))], options


def main():
    p = parser()
    args = p.parse_args()
    try:
        command, options = build_command(args)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        for key in ("train_data_path", "test_data_path", "sensitivity_profile"):
            if key in options and not Path(options[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {options[key]}. See README data preparation.")
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Choose an empty output directory; existing runs are never overwritten")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), VLLM_NO_USAGE_STATS="1")
        env.pop("RAY_ADDRESS", None)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        for prediction in sorted(args.output.resolve().glob("*/ensemble_predictions.jsonl")):
            subprocess.run([sys.executable, str(ROOT / "analysis/summarize_run.py"),
                            "--run", str(prediction.parent)], cwd=ROOT, env=env, check=True)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
