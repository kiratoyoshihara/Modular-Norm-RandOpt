#!/usr/bin/env python3
"""Search 3,000 MN candidates, then evaluate its ES-budget-matched ensembles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("search", "ensemble"), required=True)
    p.add_argument("--task", choices=("countdown", "gsm8k"), required=True)
    p.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--source-run", type=Path, help="Completed search run directory containing results.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = json.loads((ROOT / "configs/experiments.json").read_text())
    model, task = cfg["models"]["qwen-1.5b"], cfg["tasks"][args.task]
    data, output = args.data_root.resolve(), args.output.resolve()
    if args.phase == "search":
        test = "countdown/countdown_test.json" if args.task == "countdown" else task["test"]
        command = ["/usr/bin/time", "--quiet", f"--output={output / 'os_time.json'}",
                   '--format={"elapsed_sec": %e, "user_sec": %U, "system_sec": %S, "max_rss_kb": %M, "exit_status": %x}',
                   sys.executable, str(ROOT / "scripts/baselines/es_at_scale/run_mn_randopt_es_final.py")]
        options = dict(dataset=args.task, model_name=model["name"], model_revision=model["revision"],
            train_data_path=data / task["train"], test_data_path=data / test,
            perturbation_method="recursive_modular_shell_v2", radius=.16,
            mass_config=json.dumps(cfg["mass"]), sensitivity_profile=ROOT / model["profile"],
            population_size=3000, population_prefixes=3000, top_k_values=1,
            train_samples=200, max_tokens=1024, precision="bfloat16", power_iterations=8,
            num_engines=1, tp=1, cuda_devices=args.gpu, global_seed=args.seed,
            chat_template_date=cfg["chat_template_date"], experiment_dir=output / "runs",
            os_wall_time_path=output / "os_time.json")
        command.extend(item for key, value in options.items() for item in (f"--{key}", str(value)))
        command.append("--wall_clock_mode")
    else:
        if args.source_run is None:
            p.error("--phase ensemble requires --source-run")
        command = [sys.executable, str(ROOT / "scripts/baselines/es_at_scale/run_mn_randopt_es_ensemble_eval.py"),
                   "--source-run-dir", str(args.source_run.resolve()), "--output-dir", str(output),
                   "--cuda-devices", str(args.gpu)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if output.exists():
        p.error("Use a new output directory")
    if args.phase == "search":
        output.mkdir(parents=True)
    else:
        saved = json.loads((args.source_run / "results.json").read_text())
        if saved["dataset"] != args.task or saved["global_seed"] != args.seed:
            p.error("Source run task/seed differs from the requested configuration")
    env = dict(os.environ, MN_RANDOPT_DATA=str(data), CUDA_VISIBLE_DEVICES=str(args.gpu))
    env.pop("RAY_ADDRESS", None)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
