#!/usr/bin/env python3
"""Run the predeclared ES-at-Scale baseline with exact budget accounting.

This script keeps the upstream ES core intact.  A subclass replaces only the
upstream ``fit`` loop to fix its inclusive stopping condition, evaluate/save at
predeclared budgets, and emit auditable token and wall-clock accounting.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_handlers import get_dataset_handler
from utils.experiment_logging import start_console_tee, stop_console_tee
from utils.gradient_free.budget import BudgetLedger
from utils.gradient_free.protocol import (
    CALIBRATION_ITERATIONS,
    COUNTDOWN_SIGMA_GRID,
    DEFAULT_CHAT_TEMPLATE_DATE,
    DEFAULT_CHECKPOINT_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_PRECISION,
    DEFAULT_REWARD_SHAPING,
    DEFAULT_TRAIN_SAMPLES,
    FINAL_ITERATIONS,
    OFFICIAL_ES_AT_SCALE_COMMIT,
    OFFICIAL_ES_AT_SCALE_REPOSITORY,
    OFFICIAL_ES_TRAINER_SHA256,
    OFFICIAL_ES_WORKER_SHA256,
    PROTOCOL_NAME,
    alpha_for_sigma,
    get_split_spec,
    normalize_checkpoint_iterations,
    population_seeds,
    sha256_file,
    validate_phase_assignment,
    validate_split_file,
)
from utils.gradient_free.result_schema import (
    RESULT_SCHEMA_VERSION,
    validate_iteration_record,
    validate_run_manifest,
    validate_summary,
)
from utils.gradient_free.task_adapter import (
    collate_es_batch,
    identity_template,
    make_reward_function,
    prepare_task_split,
)


EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "vllm": "0.11.0",
    "ray": "2.56.1",
}
SOURCE_PATHS = (
    "scripts/baselines/es_at_scale/run_es_baseline.py",
    "utils/gradient_free/__init__.py",
    "utils/gradient_free/budget.py",
    "utils/gradient_free/protocol.py",
    "utils/gradient_free/result_schema.py",
    "utils/gradient_free/task_adapter.py",
    "utils/official_prompt_protocol.py",
    "data_handlers/base.py",
    "data_handlers/countdown.py",
    "data_handlers/gsm8k.py",
    "utils/reward_score/countdown.py",
    "utils/reward_score/gsm8k.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_manifest() -> dict[str, Any]:
    files = {path: sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS}
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"combined_sha256": combined, "files": files}


def _environment() -> dict[str, Any]:
    versions = {
        name: _package_version(name)
        for name in (
            "torch",
            "transformers",
            "vllm",
            "ray",
            "numpy",
            "datasets",
            "pandas",
            "pyarrow",
            "huggingface-hub",
        )
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repo_git_commit": _git_output("rev-parse", "HEAD"),
        "repo_git_dirty": bool(_git_output("status", "--porcelain")),
        "source_manifest": _source_manifest(),
    }


def _validate_runtime_versions(*, allow_mismatch: bool) -> dict[str, str | None]:
    actual = {name: _package_version(name) for name in EXPECTED_RUNTIME_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in EXPECTED_RUNTIME_VERSIONS.items()
        if actual[name] != expected
    }
    if mismatches and not allow_mismatch:
        details = ", ".join(
            f"{name}: expected {row['expected']}, got {row['actual']}"
            for name, row in mismatches.items()
        )
        raise RuntimeError(
            f"ES environment does not match requirements-es.lock ({details}). "
            "Use scripts/baselines/es_at_scale/bootstrap_env.sh."
        )
    return actual


def _verify_upstream(source: Path) -> dict[str, Any]:
    source = source.resolve()
    trainer_path = source / "es_at_scale/trainer/es_trainer.py"
    worker_path = source / "es_at_scale/utils/worker_extension.py"
    if not trainer_path.is_file() or not worker_path.is_file():
        raise FileNotFoundError(
            f"Pinned ES-at-Scale source not found under {source}. "
            "Run scripts/baselines/es_at_scale/bootstrap_env.sh first."
        )
    commit = _git_output("rev-parse", "HEAD", cwd=source)
    if commit != OFFICIAL_ES_AT_SCALE_COMMIT:
        raise RuntimeError(
            f"ES-at-Scale commit mismatch: expected {OFFICIAL_ES_AT_SCALE_COMMIT}, "
            f"got {commit}"
        )
    dirty = _git_output("status", "--porcelain", cwd=source)
    if dirty is None:
        raise RuntimeError(f"Could not inspect ES-at-Scale worktree: {source}")
    if dirty:
        raise RuntimeError(
            "Pinned ES-at-Scale checkout has local or untracked changes; refusing to run"
        )
    hashes = {
        "es_at_scale/trainer/es_trainer.py": sha256_file(trainer_path),
        "es_at_scale/utils/worker_extension.py": sha256_file(worker_path),
    }
    expected_hashes = {
        "es_at_scale/trainer/es_trainer.py": OFFICIAL_ES_TRAINER_SHA256,
        "es_at_scale/utils/worker_extension.py": OFFICIAL_ES_WORKER_SHA256,
    }
    if hashes != expected_hashes:
        raise RuntimeError(
            "Pinned ES-at-Scale source has local modifications; refusing to run"
        )
    return {
        "repository": OFFICIAL_ES_AT_SCALE_REPOSITORY,
        "commit": commit,
        "source_root": str(source),
        "source_sha256": hashes,
        "local_modifications": False,
    }


def _import_upstream_trainer(source: Path) -> type:
    source_text = str(source.resolve())
    sys.path.insert(0, source_text)
    # Ray actors start fresh interpreters and do not inherit driver-only
    # ``sys.path`` mutations.  Export the verified source root before ray.init
    # so worker_extension_cls can import the exact same pinned checkout.
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in inherited_pythonpath.split(os.pathsep) if part]
    if source_text not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [source_text, *pythonpath_parts]
        )
    from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer

    return EvolutionStrategiesTrainer


def _load_selection_artifact(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol",
        "task",
        "selection_split",
        "selected_sigma",
        "calibration_iterations",
        "population_size",
        "upstream_commit",
        "model",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Selection artifact is missing fields: {missing}")
    if payload["schema_version"] != "es-hparam-selection-v1":
        raise ValueError("Unknown selection artifact schema")
    if payload["protocol"] != PROTOCOL_NAME:
        raise ValueError("Selection artifact protocol mismatch")
    if payload["task"] != "countdown" or payload["selection_split"] != "validation":
        raise ValueError("Final hyperparameters must be selected on Countdown validation")
    if int(payload["calibration_iterations"]) != CALIBRATION_ITERATIONS:
        raise ValueError("Selection artifact used the wrong calibration budget")
    if int(payload["population_size"]) != DEFAULT_POPULATION_SIZE:
        raise ValueError("Selection artifact used the wrong population size")
    if payload["upstream_commit"] != OFFICIAL_ES_AT_SCALE_COMMIT:
        raise ValueError("Selection artifact upstream commit mismatch")
    sigma = float(payload["selected_sigma"])
    if sigma not in COUNTDOWN_SIGMA_GRID:
        raise ValueError(f"Selected sigma {sigma} was not in the predeclared grid")
    return payload


def _resolve_model_source(model_name: str, revision: str | None) -> tuple[str, str | None]:
    if revision is None:
        return model_name, None
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    cached_config = try_to_load_from_cache(
        repo_id=model_name,
        filename="config.json",
        revision=revision,
    )
    if isinstance(cached_config, str):
        # Keep the snapshot directory rather than following the config symlink
        # into the shared blob store.
        snapshot = Path(cached_config).parent
        has_weights = any(snapshot.glob("*.safetensors")) or any(
            snapshot.glob("pytorch_model*.bin")
        )
        if has_weights:
            return str(snapshot), revision
    local_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
        allow_patterns=(
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.model",
            "*.txt",
            "*.jinja",
        ),
    )
    return local_path, revision


def _parse_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _default_experiment_name(args: argparse.Namespace, sigma: float) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = args.model_name.rsplit("/", 1)[-1]
    return (
        f"{args.phase}-{args.task}-{model}-sigma{sigma:g}-pop{args.population_size}"
        f"-iter{args.iterations}-seed{args.seed}-{timestamp}"
    )


def _visible_gpu_count(cuda_devices: str) -> int:
    devices = [item.strip() for item in cuda_devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--cuda-devices must list at least one GPU")
    if len(set(devices)) != len(devices):
        raise ValueError("--cuda-devices contains duplicates")
    return len(devices)


def _parse_eval_file(path: Path, expected_count: int) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Evaluation output count mismatch in {path}: expected {expected_count}, "
            f"got {len(rows)}"
        )
    rewards = [float(row["reward"]) for row in rows]
    correct = sum(row.get("format") == "correct" for row in rows)
    return {
        "num_examples": len(rows),
        "num_correct": int(correct),
        "accuracy": correct / len(rows) if rows else 0.0,
        "mean_task_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "output_file": str(path),
    }


def _make_budgeted_trainer(base_class: type) -> type:
    import ray

    class BudgetedEvolutionStrategiesTrainer(base_class):
        """Official trainer with an exact, resumable, instrumented fit loop."""

        def configure_protocol(
            self,
            *,
            run_dir: Path,
            task: str,
            phase: str,
            eval_name: str,
            eval_examples: int,
            evaluation_iterations: Sequence[int],
            save_iterations: Sequence[int],
            resume_iteration: int,
            ledger: BudgetLedger,
            upstream: Mapping[str, Any],
            resume_evaluation: Mapping[str, Any] | None,
        ) -> None:
            self.protocol_run_dir = run_dir
            self.protocol_task = task
            self.protocol_phase = phase
            self.protocol_eval_name = eval_name
            self.protocol_eval_examples = eval_examples
            self.protocol_evaluation_iterations = set(
                int(v) for v in evaluation_iterations
            )
            self.protocol_save_iterations = set(int(v) for v in save_iterations)
            self.protocol_resume_iteration = int(resume_iteration)
            self.protocol_ledger = ledger
            self.protocol_upstream = dict(upstream)
            self.protocol_resume_evaluation = (
                dict(resume_evaluation) if resume_evaluation is not None else None
            )
            self.protocol_evaluations: list[dict[str, Any]] = []
            self.protocol_search_wall_sec = 0.0
            self.protocol_eval_wall_sec = 0.0
            self._metric_phase = "idle"
            self._pending_search_prompt_tokens = 0
            self._pending_search_completion_tokens = 0
            self._pending_eval_prompts = 0
            self._pending_eval_prompt_tokens = 0
            self._pending_eval_completion_tokens = 0

        def _postprocess_outputs(self, generated_text, target_text, eval=False):
            metrics = super()._postprocess_outputs(
                generated_text, target_text, eval=eval
            )
            prompt_tokens = 0
            for generated in generated_text:
                ids = getattr(generated, "prompt_token_ids", None)
                if ids is None:
                    raise RuntimeError(
                        "vLLM output did not expose prompt_token_ids; exact token "
                        "accounting is required"
                    )
                prompt_tokens += len(ids)
            completion_tokens = sum(
                int(length)
                for rollout_lengths in metrics["raw_lens_per_prompt"]
                for length in rollout_lengths
            )
            if self._metric_phase == "search":
                self._pending_search_prompt_tokens += prompt_tokens
                self._pending_search_completion_tokens += completion_tokens
            elif self._metric_phase == "evaluation":
                self._pending_eval_prompts += len(generated_text)
                self._pending_eval_prompt_tokens += prompt_tokens
                self._pending_eval_completion_tokens += completion_tokens
            return metrics

        def _save_protocol_checkpoint(self, completed_iterations: int) -> Path:
            checkpoint_dir = (
                self.protocol_run_dir
                / "checkpoints"
                / f"iteration_{completed_iterations:04d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_dir / "pytorch_model.pth"
            ray.get(
                self.engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk", args=(str(weights_path),)
                )
            )
            state = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "protocol": PROTOCOL_NAME,
                "method": "es-at-scale",
                "task": self.protocol_task,
                "phase": self.protocol_phase,
                "completed_iterations": completed_iterations,
                "next_zero_based_iteration": completed_iterations,
                "population_size": self.population_size,
                "batch_size": self.batch_size,
                "sigma": self.sigma,
                "alpha": self.alpha,
                "global_seed": self.global_seed,
                "budget": self.protocol_ledger.snapshot(),
                "upstream": self.protocol_upstream,
                "last_evaluation": (
                    self.protocol_evaluations[-1]
                    if self.protocol_evaluations
                    else None
                ),
                "weights_path": str(weights_path),
                "saved_at": _utc_now(),
            }
            _json_dump(checkpoint_dir / "state.json", state)
            return weights_path

        def _evaluate_protocol_checkpoint(
            self, completed_iterations: int
        ) -> dict[str, Any]:
            self._pending_eval_prompts = 0
            self._pending_eval_prompt_tokens = 0
            self._pending_eval_completion_tokens = 0
            self._metric_phase = "evaluation"
            started = time.perf_counter()
            # Upstream adds one to the filename. Passing n-1 yields iteration_n,
            # including n=0 -> -1 -> iteration0 for the base checkpoint.
            super().eval_step(iteration=completed_iterations - 1)
            elapsed = time.perf_counter() - started
            self.protocol_eval_wall_sec += elapsed
            self._metric_phase = "idle"
            self.protocol_ledger.add_evaluation(
                num_prompts=self._pending_eval_prompts,
                prompt_tokens=self._pending_eval_prompt_tokens,
                completion_tokens=self._pending_eval_completion_tokens,
            )
            output_path = (
                self.protocol_run_dir
                / "eval-output"
                / f"model_eval_task{self.protocol_eval_name}_iteration{completed_iterations}.json"
            )
            parsed = _parse_eval_file(output_path, self.protocol_eval_examples)
            event = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "event": "evaluation",
                "iteration": completed_iterations,
                "candidate_evaluations": (
                    completed_iterations * self.population_size
                ),
                "model_prompt_evaluations": (
                    completed_iterations * self.population_size * self.batch_size
                ),
                "split": self.protocol_eval_name,
                "elapsed_sec": elapsed,
                "prompt_tokens": self._pending_eval_prompt_tokens,
                "completion_tokens": self._pending_eval_completion_tokens,
                "budget": self.protocol_ledger.snapshot(),
                **parsed,
            }
            validate_iteration_record(event)
            _append_jsonl(self.protocol_run_dir / "evaluation_metrics.jsonl", event)
            self.protocol_evaluations.append(event)
            return event

        def fit_exact(self) -> dict[str, Any]:
            resume = self.protocol_resume_iteration
            if len(self.train_dataloader) != 1:
                raise RuntimeError(
                    "Official protocol requires exactly one fixed 200-example "
                    "training batch per ES iteration"
                )

            if resume == 0:
                self._evaluate_protocol_checkpoint(0)
            elif self.protocol_resume_evaluation is not None:
                reference = dict(self.protocol_resume_evaluation)
                reference["event"] = "evaluation_resume_reference"
                reference["resumed_reference"] = True
                validate_iteration_record(reference)
                _append_jsonl(
                    self.protocol_run_dir / "evaluation_metrics.jsonl", reference
                )
                self.protocol_evaluations.append(reference)
            else:
                # Older protocol checkpoints did not retain the evaluation
                # record, so evaluate once rather than silently losing it.
                self._evaluate_protocol_checkpoint(resume)
            previous_record: Mapping[str, Any] | None = None
            for completed in range(resume + 1, self.num_iterations + 1):
                input_text, target_text = next(iter(self.train_dataloader))
                input_text = [self.template(item) for item in input_text]
                zero_based_iteration = completed - 1
                seeds = list(
                    population_seeds(
                        self.global_seed or 42,
                        zero_based_iteration,
                        self.population_size,
                    )
                )

                self.protocol_ledger.begin_iteration(len(input_text))
                self._pending_search_prompt_tokens = 0
                self._pending_search_completion_tokens = 0
                self._metric_phase = "search"
                started = time.perf_counter()
                status = "completed"
                error = None
                try:
                    self.train_step(
                        iteration=zero_based_iteration,
                        seeds=seeds,
                        input_text=input_text,
                        target_text=target_text,
                    )
                except BaseException as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    elapsed = time.perf_counter() - started
                    self.protocol_search_wall_sec += elapsed
                    self.protocol_ledger.add_search_tokens(
                        prompt_tokens=self._pending_search_prompt_tokens,
                        completion_tokens=self._pending_search_completion_tokens,
                    )
                    if status == "completed":
                        self.protocol_ledger.complete_iteration(len(input_text))
                    self._metric_phase = "idle"
                    record = {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "event": "search_iteration",
                        "status": status,
                        "iteration": completed,
                        "zero_based_iteration": zero_based_iteration,
                        "candidate_evaluations": (
                            self.protocol_ledger.completed_candidate_evaluations
                        ),
                        "model_prompt_evaluations": (
                            self.protocol_ledger.completed_model_prompt_evaluations
                        ),
                        "attempted_candidate_evaluations": (
                            self.protocol_ledger.attempted_candidate_evaluations
                        ),
                        "attempted_model_prompt_evaluations": (
                            self.protocol_ledger.attempted_model_prompt_evaluations
                        ),
                        "population_seeds": [int(seed) for seed in seeds],
                        "elapsed_sec": elapsed,
                        "iteration_prompt_tokens": self._pending_search_prompt_tokens,
                        "iteration_completion_tokens": (
                            self._pending_search_completion_tokens
                        ),
                        "error": error,
                        "budget": self.protocol_ledger.snapshot(),
                    }
                    validate_iteration_record(record, previous_record)
                    _append_jsonl(
                        self.protocol_run_dir / "iteration_metrics.jsonl", record
                    )
                    previous_record = record

                self.protocol_ledger.assert_consistent()
                if completed in self.protocol_evaluation_iterations:
                    self._evaluate_protocol_checkpoint(completed)
                if completed in self.protocol_save_iterations:
                    self._save_protocol_checkpoint(completed)

            return {
                "completed_iterations": self.protocol_ledger.completed_iterations,
                "budget": self.protocol_ledger.snapshot(),
                "evaluations": self.protocol_evaluations,
                "search_wall_clock_sec": self.protocol_search_wall_sec,
                "evaluation_wall_clock_sec": self.protocol_eval_wall_sec,
            }

        def shutdown_protocol(self) -> None:
            try:
                self.cleanup()
            finally:
                try:
                    self.mp_pool.close()
                    self.mp_pool.join()
                except Exception:
                    pass
                try:
                    ray.shutdown()
                except Exception:
                    pass

    return BudgetedEvolutionStrategiesTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("countdown", "gsm8k"), required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--phase", choices=("calibration", "final", "smoke"), required=True
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--mini-batch-size", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--reward-shaping", default=DEFAULT_REWARD_SHAPING)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--num-engines", type=int)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--evaluation-iterations",
        "--checkpoint-iterations",
        dest="evaluation_iterations",
        help="Comma-separated held-out evaluation iterations",
    )
    parser.add_argument(
        "--save-iterations",
        help="Comma-separated full-weight checkpoint iterations",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-iteration", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("logs/gradient-free/es-at-scale")
    )
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--os-wall-time-path",
        type=Path,
        help=(
            "Path populated by an external GNU time process after this runner exits"
        ),
    )
    parser.add_argument(
        "--es-at-scale-source",
        type=Path,
        default=Path(
            os.environ.get("ES_AT_SCALE_SOURCE", REPO_ROOT / "external/es-at-scale")
        ),
    )
    parser.add_argument("--chat-template-date", default=DEFAULT_CHAT_TEMPLATE_DATE)
    parser.add_argument("--reward-function-timeout", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--allow-protocol-override", action="store_true")
    return parser


def _resolve_protocol(args: argparse.Namespace) -> dict[str, Any]:
    eval_role = validate_phase_assignment(args.task, args.phase, args.seed)
    expected_iterations = {
        "calibration": CALIBRATION_ITERATIONS,
        "final": FINAL_ITERATIONS,
        "smoke": 1,
    }[args.phase]
    iterations = expected_iterations if args.iterations is None else args.iterations
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if (
        args.phase != "smoke"
        and iterations != expected_iterations
        and not args.allow_protocol_override
    ):
        raise ValueError(
            f"{args.phase} requires {expected_iterations} iterations, got {iterations}"
        )
    if (
        args.phase != "smoke"
        and args.population_size != DEFAULT_POPULATION_SIZE
        and not args.allow_protocol_override
    ):
        raise ValueError(
            f"Official protocol requires population {DEFAULT_POPULATION_SIZE}"
        )
    if (
        args.phase != "smoke"
        and args.train_samples != DEFAULT_TRAIN_SAMPLES
        and not args.allow_protocol_override
    ):
        raise ValueError(f"Official protocol requires {DEFAULT_TRAIN_SAMPLES} prompts")
    if args.phase != "smoke" and args.eval_samples is not None:
        raise ValueError("Calibration/final evaluation must use the complete held-out split")
    if args.precision != "bfloat16" and not args.allow_protocol_override:
        raise ValueError("Official protocol requires bfloat16")
    if args.reward_shaping != "z-scores":
        raise ValueError("Pinned upstream supports only z-scores reward shaping")

    selection = None
    if args.phase == "final":
        if args.selection_artifact is None:
            raise ValueError("Final runs require --selection-artifact")
        selection = _load_selection_artifact(args.selection_artifact.resolve())
        if selection["model"]["requested_name"] != args.model_name:
            raise ValueError("Final model differs from the Countdown calibration model")
        if selection["model"]["requested_revision"] != args.model_revision:
            raise ValueError(
                "Final model revision differs from the Countdown calibration revision"
            )
        selected_sigma = float(selection["selected_sigma"])
        if args.sigma is not None and args.sigma != selected_sigma:
            raise ValueError(
                f"--sigma {args.sigma} differs from frozen selected sigma {selected_sigma}"
            )
        sigma = selected_sigma
    elif args.phase == "calibration":
        if args.selection_artifact is not None:
            raise ValueError("Calibration must not consume a selection artifact")
        if args.sigma is None or args.sigma not in COUNTDOWN_SIGMA_GRID:
            raise ValueError(
                f"Calibration sigma must be one of {COUNTDOWN_SIGMA_GRID}"
            )
        sigma = float(args.sigma)
    else:
        sigma = 0.001 if args.sigma is None else float(args.sigma)

    if args.resume_iteration < 0 or args.resume_iteration > iterations:
        raise ValueError("resume iteration must lie in [0, iterations]")
    if bool(args.checkpoint) != bool(args.resume_iteration):
        raise ValueError("--checkpoint and a positive --resume-iteration are required together")
    if args.phase == "final" and args.checkpoint is not None:
        state_path = args.checkpoint.parent / "state.json"
        if not state_path.is_file():
            raise FileNotFoundError("Final resume requires the checkpoint state.json")

    default_evaluations = (
        DEFAULT_CHECKPOINT_ITERATIONS if args.phase == "final" else (iterations,)
    )
    evaluation_values = (
        default_evaluations
        if args.evaluation_iterations is None
        else _parse_int_csv(args.evaluation_iterations)
    )
    evaluation_iterations = normalize_checkpoint_iterations(
        (*evaluation_values, iterations),
        total_iterations=iterations,
    )
    default_saves = (10, 30, 100) if args.phase == "final" else ()
    if args.phase == "smoke":
        default_saves = (iterations,)
    save_values = (
        default_saves
        if args.save_iterations is None
        else _parse_int_csv(args.save_iterations)
    )
    save_iterations = normalize_checkpoint_iterations(
        save_values,
        total_iterations=iterations,
    )
    return {
        "eval_role": eval_role,
        "iterations": iterations,
        "sigma": sigma,
        "alpha": alpha_for_sigma(sigma),
        "selection": selection,
        "evaluation_iterations": evaluation_iterations,
        "save_iterations": save_iterations,
    }


def _load_resume_state(
    args: argparse.Namespace,
    *,
    population_size: int,
    prompts: int,
    task: str,
    phase: str,
    sigma: float,
    alpha: float,
) -> tuple[BudgetLedger, Mapping[str, Any] | None]:
    if args.checkpoint is None:
        return BudgetLedger(population_size, prompts), None
    state_path = args.checkpoint.parent / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state["completed_iterations"]) != args.resume_iteration:
        raise ValueError("resume iteration disagrees with checkpoint state")
    if int(state["population_size"]) != population_size:
        raise ValueError("resume checkpoint population mismatch")
    if int(state["batch_size"]) != prompts:
        raise ValueError("resume checkpoint train batch mismatch")
    if state.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema mismatch")
    if state.get("protocol") != PROTOCOL_NAME or state.get("method") != "es-at-scale":
        raise ValueError("resume checkpoint protocol mismatch")
    if state.get("task") != task or state.get("phase") != phase:
        raise ValueError("resume checkpoint task/phase mismatch")
    if float(state["sigma"]) != sigma or float(state["alpha"]) != alpha:
        raise ValueError("resume checkpoint hyperparameter mismatch")
    if int(state["global_seed"]) != args.seed:
        raise ValueError("resume checkpoint global seed mismatch")
    if state["upstream"]["commit"] != OFFICIAL_ES_AT_SCALE_COMMIT:
        raise ValueError("resume checkpoint upstream commit mismatch")
    if Path(state["weights_path"]).resolve() != args.checkpoint.resolve():
        raise ValueError("resume checkpoint path disagrees with state.json")
    ledger = BudgetLedger.from_snapshot(state["budget"])
    ledger.assert_consistent()
    return ledger, state.get("last_evaluation")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = _resolve_protocol(args)
    args.iterations = protocol["iterations"]
    sigma = protocol["sigma"]
    alpha = protocol["alpha"]

    train_spec = get_split_spec(args.task, "train")
    eval_spec = get_split_spec(args.task, protocol["eval_role"])
    train_spec = replace(train_spec, relative_path=str(args.data_root.resolve() / Path(train_spec.relative_path).relative_to("data")))
    eval_spec = replace(eval_spec, relative_path=str(args.data_root.resolve() / Path(eval_spec.relative_path).relative_to("data")))
    train_path = validate_split_file(REPO_ROOT, train_spec)
    eval_path = validate_split_file(REPO_ROOT, eval_spec)

    strict_environment = args.phase != "smoke" or not args.allow_environment_mismatch
    versions = _validate_runtime_versions(
        allow_mismatch=not strict_environment or args.allow_environment_mismatch
    )
    upstream = _verify_upstream(args.es_at_scale_source)

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    model_source, requested_revision = _resolve_model_source(
        args.model_name, args.model_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    resolved_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    train_split = prepare_task_split(
        task=args.task,
        role="train",
        path=train_path,
        tokenizer=tokenizer,
        model_name=args.model_name,
        chat_template_date=args.chat_template_date,
        max_samples=args.train_samples,
        expected=train_spec,
    )
    eval_split = prepare_task_split(
        task=args.task,
        role=protocol["eval_role"],
        path=eval_path,
        tokenizer=tokenizer,
        model_name=args.model_name,
        chat_template_date=args.chat_template_date,
        max_samples=args.eval_samples,
        expected=eval_spec,
    )
    if len(train_split) != args.train_samples:
        raise ValueError(
            f"Requested {args.train_samples} training examples, got {len(train_split)}"
        )

    gpu_count = _visible_gpu_count(args.cuda_devices)
    if gpu_count % args.tp:
        raise ValueError("visible GPU count must be divisible by tensor parallel size")
    num_engines = args.num_engines or (gpu_count // args.tp)
    if num_engines * args.tp > gpu_count:
        raise ValueError("requested engines use more GPUs than --cuda-devices provides")

    experiment_name = args.experiment_name or _default_experiment_name(args, sigma)
    output_root = (REPO_ROOT / args.output_root).resolve()
    run_dir = output_root / experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    start_console_tee(str(run_dir))

    environment = _environment()
    environment["packages"].update(versions)
    environment["allocated_gpus"] = num_engines * args.tp
    environment["num_engines"] = num_engines
    environment["tensor_parallel_size"] = args.tp
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "method": "es-at-scale",
        "phase": args.phase,
        "task": args.task,
        "model": {
            "requested_name": args.model_name,
            "runtime_source": model_source,
            "requested_revision": requested_revision,
            "resolved_tokenizer_revision": resolved_revision,
        },
        "seed": args.seed,
        "hyperparameters": {
            "sigma": sigma,
            "alpha": alpha,
            "population_size": args.population_size,
            "iterations": args.iterations,
            "batch_size": len(train_split),
            "mini_batch_size": args.mini_batch_size,
            "max_tokens": args.max_tokens,
            "reward_shaping": args.reward_shaping,
            "temperature": 0.0,
            "top_p": 1.0,
            "precision": args.precision,
            "evaluation_iterations": list(protocol["evaluation_iterations"]),
            "save_iterations": list(protocol["save_iterations"]),
        },
        "budget_definition": {
            "candidate_evaluations": "population_size * completed_iterations",
            "model_prompt_evaluations": (
                "population_size * completed_iterations * training_prompts"
            ),
            "held_out_evaluation_excluded_from_search_budget": True,
        },
        "splits": {
            "train": {
                "role": "optimization",
                "path": str(train_path),
                "sha256": train_split.source_sha256,
                "num_examples": len(train_split),
            },
            protocol["eval_role"]: {
                "role": "hyperparameter_selection"
                if args.phase == "calibration"
                else "report_only",
                "path": str(eval_path),
                "sha256": eval_split.source_sha256,
                "num_examples": len(eval_split),
            },
        },
        "prompt_tokenization": {
            "train": train_split.prompt_metadata,
            protocol["eval_role"]: eval_split.prompt_metadata,
        },
        "selection_artifact": (
            {
                "path": str(args.selection_artifact.resolve()),
                "sha256": sha256_file(args.selection_artifact.resolve()),
            }
            if args.selection_artifact is not None
            else None
        ),
        "upstream": upstream,
        "environment": environment,
        "external_wall_clock": (
            {
                "clock": "GNU /usr/bin/time",
                "path": str(args.os_wall_time_path.resolve()),
                "record_available_after_process_exit": True,
            }
            if args.os_wall_time_path is not None
            else None
        ),
        "protocol_override": bool(args.allow_protocol_override),
        "created_at": _utc_now(),
        "run_dir": str(run_dir),
    }
    validate_run_manifest(manifest)
    _json_dump(run_dir / "run_manifest.json", manifest)
    _json_dump(run_dir / "args.json", vars(args))
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    (run_dir / "environment.freeze.txt").write_text(freeze, encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(
        f"Protocol: phase={args.phase}, task={args.task}, seed={args.seed}, "
        f"sigma={sigma}, alpha={alpha}, P={args.population_size}, "
        f"T={args.iterations}"
    )
    print(
        f"Search budget: {args.population_size * args.iterations} candidate models, "
        f"{args.population_size * args.iterations * len(train_split)} model-prompts"
    )
    if args.preflight_only:
        summary = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "completed",
            "task": args.task,
            "phase": args.phase,
            "seed": args.seed,
            "completed_iterations": 0,
            "budget": BudgetLedger(args.population_size, len(train_split)).snapshot(),
            "evaluations": [],
            "wall_clock_sec": 0.0,
            "external_wall_clock_path": (
                str(args.os_wall_time_path.resolve())
                if args.os_wall_time_path is not None
                else None
            ),
            "preflight_only": True,
        }
        validate_summary(summary)
        _json_dump(run_dir / "summary.json", summary)
        stop_console_tee()
        return 0

    base_class = _import_upstream_trainer(args.es_at_scale_source)
    train_loader = DataLoader(
        train_split,
        batch_size=len(train_split),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_es_batch,
    )
    eval_loader = DataLoader(
        eval_split,
        batch_size=min(args.mini_batch_size, len(eval_split)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_es_batch,
    )
    ledger, resume_evaluation = _load_resume_state(
        args,
        population_size=args.population_size,
        prompts=len(train_split),
        task=args.task,
        phase=args.phase,
        sigma=sigma,
        alpha=alpha,
    )
    trainer_class = _make_budgeted_trainer(base_class)
    trainer = None
    total_started = time.perf_counter()
    status = "failed"
    run_result: dict[str, Any] = {}
    failure: dict[str, Any] | None = None
    try:
        trainer = trainer_class(
            model_name=model_source,
            checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
            sigma=sigma,
            alpha=alpha,
            population_size=args.population_size,
            reward_shaping=args.reward_shaping,
            num_iterations=args.iterations,
            max_tokens=args.max_tokens,
            batch_size=len(train_split),
            mini_batch_size=args.mini_batch_size,
            reward_function=make_reward_function(args.task),
            template_function=identity_template,
            train_dataloader=train_loader,
            eval_dataloader_dict={protocol["eval_role"]: eval_loader},
            eval_freq=max(1, args.iterations + 1),
            n_vllm_engines=num_engines,
            n_gpu_per_vllm_engine=args.tp,
            logging="none",
            global_seed=args.seed,
            output_directory=str(output_root),
            save_best_models=False,
            experiment_name=experiment_name,
            wandb_project=None,
            reward_function_timeout=args.reward_function_timeout,
            use_gpus=args.cuda_devices,
        )
        trainer.configure_protocol(
            run_dir=run_dir,
            task=args.task,
            phase=args.phase,
            eval_name=protocol["eval_role"],
            eval_examples=len(eval_split),
            evaluation_iterations=protocol["evaluation_iterations"],
            save_iterations=protocol["save_iterations"],
            resume_iteration=args.resume_iteration,
            ledger=ledger,
            upstream=upstream,
            resume_evaluation=resume_evaluation,
        )
        run_result = trainer.fit_exact()
        status = "completed"
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        total_wall = time.perf_counter() - total_started
        if trainer is not None:
            try:
                trainer.shutdown_protocol()
            except Exception as cleanup_error:
                print(f"Cleanup warning: {cleanup_error}", file=sys.stderr)
        summary = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": status,
            "task": args.task,
            "phase": args.phase,
            "seed": args.seed,
            "completed_iterations": ledger.completed_iterations,
            "budget": ledger.snapshot(),
            "evaluations": run_result.get("evaluations", []),
            "wall_clock_sec": total_wall,
            "search_wall_clock_sec": run_result.get("search_wall_clock_sec", 0.0),
            "evaluation_wall_clock_sec": run_result.get(
                "evaluation_wall_clock_sec", 0.0
            ),
            "allocated_gpus": num_engines * args.tp,
            "gpu_hours": total_wall * (num_engines * args.tp) / 3600.0,
            "external_wall_clock_path": (
                str(args.os_wall_time_path.resolve())
                if args.os_wall_time_path is not None
                else None
            ),
            "failure": failure,
            "finished_at": _utc_now(),
        }
        validate_summary(summary)
        _json_dump(run_dir / "summary.json", summary)
        stop_console_tee()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
