#!/usr/bin/env python3
"""Train a single MeZO or task-adapted ZO-Finetuner run, without a scheduler."""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import common as io


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("mezo", "zo-finetuner"))
    p.add_argument("--task", required=True, choices=("countdown", "gsm8k"))
    p.add_argument("--phase", required=True, choices=("smoke", "calibration", "final"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/iterative.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, help="Default: the published selected learning rate.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--generator", type=Path, default=ROOT / "artifacts/zo_generator.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def configuration(args):
    cfg = copy.deepcopy(io.read(args.config))
    lr = cfg["learning_rate"] if args.learning_rate is None else args.learning_rate
    if not math.isfinite(lr) or lr <= 0 or args.gpu < 0:
        raise ValueError("Learning rate must be finite and positive, and GPU index nonnegative")
    if args.phase == "calibration":
        if args.seed != cfg["calibration_seed"] or args.task != "countdown":
            raise ValueError("Calibration uses Countdown and seed 39")
        if args.learning_rate not in cfg["learning_rate_grids"][args.method]:
            raise ValueError("Choose --learning-rate from the method's calibration grid")
    if args.phase == "final" and args.seed not in cfg["final_seeds"]:
        raise ValueError("Final seeds are 42, 43, and 44")
    cfg["learning_rate"] = lr
    cfg["steps"] = 1 if args.phase == "smoke" else cfg[f"{args.phase}_steps"]
    cfg["smoke"] = args.phase == "smoke"
    if cfg["smoke"]:
        cfg["max_new_tokens"] = 32
    return cfg


def run(args, cfg):
    import torch
    from huggingface_hub import snapshot_download
    from baselines.zo import Generation, amplitudes, step as zo_step, weight_metrics
    from baselines.mezo import step as mezo_step
    from baselines.optimizers import PerturbationGenerator

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; completed or partial runs are never overwritten")
    if args.method == "zo-finetuner" and not args.generator.is_file():
        raise FileNotFoundError(f"Missing generator: {args.generator}")
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    model, tokenizer = io.load_model(cfg, getattr(torch, cfg["precision"]))
    model.requires_grad_(False)
    parameters = list(model.named_parameters())
    generator = None
    if args.method == "zo-finetuner":
        saved = torch.load(args.generator, map_location="cpu", weights_only=True)
        if saved["layout"] != [[name, list(p.shape)] for name, p in parameters]:
            raise ValueError("Generator/model parameter layout mismatch")
        generator = PerturbationGenerator(parameters, cfg["logical_batch_size"]).to(model.device, model.dtype)
        generator.load_state_dict(saved["networks"])
        generator.eval().requires_grad_(False)
        generator.reset_history()
        del saved
    snapshot = snapshot_download(cfg["model"], revision=cfg["revision"], local_files_only=True)
    backend = Generation(model, tokenizer, cfg, snapshot, cfg["precision"])
    counts, data = {}, {}
    best, initial, terminal, done = None, None, None, 0
    io.write(output / "configuration.json", dict(config=cfg, method=args.method, task=args.task,
        seed=args.seed, generator_sha256=io.digest(args.generator) if generator else None))

    def evaluate(role, step, save_predictions=False):
        if role == "test" and args.phase not in ("final", "smoke"):
            raise ValueError("Calibration cannot evaluate the final test")
        if role not in data:
            data[role] = io.load_split(args.task, role, tokenizer, cfg, args.data_root)
        handler, rows, ids = data[role]
        items = []
        for offset in range(0, len(rows), 200):
            _, chunk = backend.generate_scores(handler, rows[offset:offset+200], ids[offset:offset+200], verify=True)
            for item in chunk:
                item["index"] += offset
                items.append(item)
                if save_predictions:
                    io.append(output / f"{role}-step{step}.jsonl", item)
        result = dict(role=role, step=step, examples=len(items),
            accuracy=mean(item["correct"] for item in items), mean_reward=mean(item["reward"] for item in items),
            generated_tokens=sum(item["generated_tokens"] for item in items))
        counts[role] = counts.get(role, 0) + len(items)
        io.append(output / "evaluations.jsonl", result)
        return result

    def checkpoint(name):
        io.save_checkpoint(output / name, dict(model=model.state_dict(), step=done,
            counts=dict(counts), best_development=best,
            generator_dynamics=generator.dynamics() if generator else None))

    def development():
        nonlocal best, initial, terminal
        weight_metrics(parameters)
        value = evaluate("development", done, True)
        if initial is None:
            initial = value
        terminal = value
        if best is None or io.checkpoint_key(value) > io.checkpoint_key(best):
            best = value
            checkpoint("best.pt")
        checkpoint("latest.pt")

    try:
        development()
        for index in range(cfg["steps"]):
            objective = lambda: -evaluate("train", index+1)["mean_reward"]
            seed = args.seed * 1_000_003 + index
            if generator is None:
                positive, negative, projected = mezo_step(parameters, seed, cfg["epsilon"], cfg["learning_rate"], objective)
            else:
                raw, normalization = amplitudes(generator, parameters)
                positive, negative, projected = zo_step(parameters, seed, raw, normalization,
                    cfg["epsilon"], cfg["learning_rate"], objective)
                generator.history = (positive, negative)
            done = index + 1
            io.append(output / "steps.jsonl", dict(step=done, positive=positive, negative=negative, projected_gradient=projected))
            if done % cfg["development_interval"] == 0 or done == cfg["steps"]:
                development()
            print(f"{args.method} {args.task} seed{args.seed}: {done}/{cfg['steps']}", flush=True)
        expected = cfg["steps"] * 2 * (2 if cfg["smoke"] else cfg["logical_batch_size"])
        if counts["train"] != expected:
            raise ValueError("Training evaluation budget mismatch")
        result = dict(method=args.method, task=args.task, seed=args.seed, steps=done,
            learning_rate=cfg["learning_rate"], best_development=best,
            initial_development=initial, terminal_development=terminal,
            stable=terminal["accuracy"] + cfg["max_terminal_development_accuracy_drop"] + 1e-12 >= initial["accuracy"],
            query_counts=counts, inference_models=1)
        if args.phase in ("final", "smoke"):
            state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            del state
            result["test"] = evaluate("test", best["step"], True)
        result.update(status="complete", smoke=cfg["smoke"], wall_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), finished_at=io.now())
        io.write(output / "summary.json", result)
    finally:
        backend.close()


def main():
    p = parser()
    args = p.parse_args()
    try:
        cfg = configuration(args)
        if args.dry_run:
            import json
            print(json.dumps(dict(method=args.method, task=args.task, seed=args.seed, config=cfg), indent=2))
            return
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), VLLM_ENABLE_V1_MULTIPROCESSING="0",
                          TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1")
        run(args, cfg)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
