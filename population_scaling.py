#!/usr/bin/env python3
"""Population-scaling evaluation for Modular-Norm RandOpt.

This script generates one population of size N_max, then evaluates nested
population prefixes without regenerating candidates.  For each prefix, it
selects the top-K perturbations by train reward and evaluates their majority-
vote ensemble on the validation set.

Expected repository support:
  * utils/worker_extn.py exposes perturb_self_weights / restore_self_weights
    with arguments (seed, radius, negate, method, mass_config,
    power_iterations).
  * utils/perturbation_norms.py contains the perturbation methods used by the
    Modular-Norm RandOpt implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from core import cleanup_engines, launch_engines
from data_handlers import get_dataset_handler, list_datasets
from utils.experiment_logging import start_console_tee, stop_console_tee
from utils.distance_match_artifact import validate_distance_match_artifact
from utils.functional_displacement import (
    aggregate_candidate_metrics,
    radius_objective,
    select_radius,
)
from utils.official_randopt_protocol import (
    CANDIDATE_SEED_SCHEME,
    PARAMETER_NOISE_SCHEME,
    WEIGHT_RESTORE_SCHEME,
    build_candidate_seeds,
)
from utils.official_randopt_provenance import source_manifest
from utils.official_prompt_protocol import (
    DEFAULT_CHAT_TEMPLATE_DATE,
    encode_rendered_prompts,
    prompt_manifest,
    render_prompts,
)
from utils.perturbation_norms import (
    SUPPORTED_PERTURBATION_METHODS,
    load_mass_config,
    load_sensitivity_profile,
    sensitivity_profile_fingerprint,
)
from utils.wall_clock import (
    WALL_CLOCK_PROTOCOL,
    validate_wall_clock_configuration,
    wall_clock_method,
)


CandidateRecord = Dict[str, Any]
PerturbationKey = Tuple[int, float]
REPO_ROOT = Path(__file__).resolve().parent


def _parse_int_csv(value: str, name: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return sorted(set(parsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Population-scaling evaluation for norm-aware RandOpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="countdown", choices=list_datasets())
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face model revision",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--chat_template_date",
        type=str,
        default=DEFAULT_CHAT_TEMPLATE_DATE,
        help="Fixed date passed to model chat templates",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Evaluate the unperturbed base model and stop before candidate sampling",
    )

    parser.add_argument(
        "--perturbation_method",
        type=str,
        required=True,
        choices=SUPPORTED_PERTURBATION_METHODS,
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Fixed perturbation radius for this run",
    )
    parser.add_argument(
        "--mass_config",
        type=str,
        default=None,
        help="JSON object or path to a JSON mass configuration",
    )
    parser.add_argument(
        "--sensitivity_profile",
        type=str,
        default=None,
        help="JSON object/path produced by calibrate_qwen_sensitivities.py (required for V2)",
    )
    parser.add_argument("--power_iterations", type=int, default=8)
    parser.add_argument(
        "--radius_selection_artifact",
        type=str,
        default=None,
        help=(
            "Functional-displacement JSON used only to choose the Modular-Shell "
            "radius for model transfer"
        ),
    )
    parser.add_argument(
        "--distance_match_artifact",
        type=str,
        default=None,
        help=(
            "Accuracy-blind matched-scale JSON produced by "
            "scripts/distance_match/select_matched_scales.py"
        ),
    )
    parser.add_argument(
        "--distance_match_target",
        type=str,
        choices=("low", "reference", "high"),
        default=None,
        help="Matched functional-distance target used by this run",
    )

    parser.add_argument("--population_size", type=int, default=300)
    parser.add_argument(
        "--population_prefixes",
        type=str,
        default="25,50,100,200,300",
        help="Nested population sizes evaluated from one generated population",
    )
    parser.add_argument(
        "--top_k_values",
        type=str,
        default="10,25",
        help="Absolute ensemble sizes, not population ratios",
    )

    parser.add_argument("--num_engines", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cuda_devices", type=str, default="0")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--experiment_dir", type=str, default="logs/population-scaling")
    parser.add_argument(
        "--wall_clock_mode",
        action="store_true",
        help=(
            "Enable the canonical synchronized K=25 wall-clock protocol. This "
            "requires a single independently generated population, not a prefix "
            "of a larger run."
        ),
    )
    parser.add_argument(
        "--os_wall_time_path",
        type=str,
        default=None,
        help=(
            "Path where the shell launcher stores /usr/bin/time output; recorded "
            "in wall_clock.json for validation"
        ),
    )
    parser.add_argument(
        "--skip_prediction_files",
        action="store_true",
        help="Do not save per-expert and per-ensemble validation predictions",
    )

    args = parser.parse_args()
    args.population_prefix_list = _parse_int_csv(
        args.population_prefixes, "population_prefixes"
    )
    args.top_k_list = _parse_int_csv(args.top_k_values, "top_k_values")

    if args.population_size <= 0:
        parser.error("--population_size must be positive")
    if args.train_samples <= 0:
        parser.error("--train_samples must be positive")
    if not math.isfinite(args.radius) or args.radius < 0.0:
        parser.error("--radius must be finite and non-negative")
    if args.power_iterations < 1:
        parser.error("--power_iterations must be at least 1")
    if args.num_engines < 1 or args.tp < 1:
        parser.error("--num_engines and --tp must be positive")
    if args.population_prefix_list[-1] != args.population_size:
        parser.error(
            "The largest --population_prefixes value must equal --population_size "
            "so that every generated candidate is used."
        )
    if any(prefix > args.population_size for prefix in args.population_prefix_list):
        parser.error("population prefixes cannot exceed population_size")
    if max(args.top_k_list) > min(args.population_prefix_list):
        parser.error(
            "Every top-K value must be no larger than the smallest population prefix"
        )

    args.mass_config = load_mass_config(args.mass_config)
    args.sensitivity_profile = load_sensitivity_profile(args.sensitivity_profile)
    if (
        args.perturbation_method == "recursive_modular_shell_v2"
        and args.sensitivity_profile is None
    ):
        parser.error(
            "--sensitivity_profile is required for recursive_modular_shell_v2"
        )
    if (
        args.radius_selection_artifact is not None
        and args.perturbation_method != "recursive_modular_shell_v2"
    ):
        parser.error(
            "--radius_selection_artifact is only valid for Modular-Shell model transfer"
        )
    if (args.distance_match_artifact is None) != (args.distance_match_target is None):
        parser.error(
            "--distance_match_artifact and --distance_match_target must be supplied together"
        )
    if (
        args.distance_match_artifact is not None
        and args.perturbation_method
        not in {"isotropic", "recursive_modular_shell_v2"}
    ):
        parser.error(
            "distance matching supports only isotropic and recursive_modular_shell_v2"
        )
    if (
        args.distance_match_artifact is not None
        and args.radius_selection_artifact is not None
    ):
        parser.error(
            "distance matching and model-transfer radius selection are mutually exclusive"
        )
    if args.distance_match_artifact is not None and args.base_only:
        parser.error("distance matching is not valid with --base_only")
    if args.os_wall_time_path is not None and not args.wall_clock_mode:
        parser.error("--os_wall_time_path is only valid with --wall_clock_mode")
    if args.wall_clock_mode:
        if (
            args.radius_selection_artifact is not None
            or args.distance_match_artifact is not None
        ):
            parser.error(
                "Wall-clock mode uses the fixed canonical radii and cannot load "
                "a radius-selection or distance-match artifact"
            )
        if args.num_engines * args.tp != _visible_gpu_count(args.cuda_devices):
            parser.error(
                "Wall-clock mode requires num_engines * tp to equal the number "
                "of listed CUDA devices"
            )
        try:
            validate_wall_clock_configuration(
                dataset=args.dataset,
                model_name=args.model_name,
                perturbation_method=args.perturbation_method,
                radius=args.radius,
                population_size=args.population_size,
                population_prefixes=args.population_prefix_list,
                top_k_values=args.top_k_list,
                global_seed=args.global_seed,
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                max_tokens=args.max_tokens,
                precision=args.precision,
                base_only=args.base_only,
            )
        except ValueError as exc:
            parser.error(str(exc))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.global_seed)

    return args


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value).strip("-_")


def create_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{_safe_name(args.dataset)}_{_safe_name(args.perturbation_method)}_"
        f"seed{args.global_seed}_{timestamp}"
    )
    run_dir = Path(args.experiment_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path_fingerprint(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
    elif target.is_dir():
        for child in sorted(item for item in target.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(target)).encode("utf-8"))
            digest.update(child.read_bytes())
    else:
        return "missing"
    return digest.hexdigest()


def load_and_validate_radius_artifact(
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    """Validate that J(r), and only J(r), supplied the model-transfer radius."""

    if args.radius_selection_artifact is None:
        return None
    path = Path(args.radius_selection_artifact)
    if not path.is_file():
        raise FileNotFoundError(f"Missing radius-selection artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "selection_uses_accuracy": False,
        "execution_protocol": "official_randopt_j_v1",
        "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
        "noise_scheme": PARAMETER_NOISE_SCHEME,
        "weight_restore_scheme": WEIGHT_RESTORE_SCHEME,
        "per_candidate_exact_reset": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Radius artifact {key!r} mismatch: "
                f"actual={payload.get(key)!r}, expected={expected!r}"
            )
    if int(payload.get("candidate_pool_size", -1)) != args.population_size:
        raise ValueError(
            "Radius artifact candidate pool must equal the population Nmax"
        )
    if int(payload.get("global_seed", -1)) != 42:
        raise ValueError("Radius artifact must use the fixed calibration seed 42")
    if int(payload.get("candidate_prefix", -1)) != 25:
        raise ValueError("Radius artifact must use the fixed N=25 prefix")
    candidate_pool = [int(seed) for seed in payload.get("candidate_pool", [])]
    expected_pool = build_candidate_seeds(
        int(payload.get("global_seed", -1)), args.population_size
    )
    if candidate_pool != expected_pool:
        raise ValueError("Radius artifact candidate pool is not official RandOpt RNG output")
    prefix_size = int(payload.get("candidate_prefix", -1))
    if payload.get("candidate_seeds") != candidate_pool[:prefix_size]:
        raise ValueError("Radius artifact candidate prefix does not match its Nmax pool")
    model = payload.get("model", {})
    if model.get("name") != args.model_name:
        raise ValueError("Radius artifact model does not match --model_name")
    resolved_revision = model.get("resolved_commit_hash")
    if args.model_revision and resolved_revision != args.model_revision:
        raise ValueError("--model_revision differs from the radius artifact")
    if resolved_revision:
        args.model_revision = str(resolved_revision)
    selected_radius = float(payload["selected_radius"])
    if not math.isclose(selected_radius, float(args.radius), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"--radius={args.radius} differs from J(r) selection {selected_radius}"
        )
    if sensitivity_profile_fingerprint(args.sensitivity_profile) != payload.get(
        "profile_fingerprint"
    ):
        raise ValueError("Radius artifact sensitivity profile does not match this run")
    if args.mass_config != payload.get("mass_config"):
        raise ValueError("Radius artifact mass_config does not match this run")
    if not math.isclose(
        float(payload.get("isotropic", {}).get("sigma", float("nan"))),
        0.0005,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Radius artifact must use RandOpt sigma=0.0005")
    expected_radii = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    actual_radii = [float(row["radius"]) for row in payload.get("modular_radii", [])]
    if actual_radii != expected_radii:
        raise ValueError(f"Radius artifact grid mismatch: {actual_radii}")
    dataset = payload.get("dataset", {})
    if dataset.get("name") != "countdown":
        raise ValueError("Model-transfer radius must be selected on Countdown")
    source_train_path = dataset.get("train_data_path")
    if not source_train_path or _path_fingerprint(source_train_path) != dataset.get(
        "train_data_sha256"
    ):
        raise ValueError("Radius-selection source data is missing or has changed")
    if payload.get("source_manifest") != source_manifest(REPO_ROOT):
        raise ValueError(
            "Source differs from the J(r) measurement; regenerate the radius artifact"
        )

    isotropic = payload.get("isotropic", {})
    isotropic_candidates = isotropic.get("candidates", [])
    if [int(row.get("seed", -1)) for row in isotropic_candidates] != candidate_pool[:prefix_size]:
        raise ValueError("Radius artifact isotropic candidates do not use the fixed prefix")
    if any(
        row.get("method") != "isotropic"
        or not math.isclose(
            float(row.get("radius", float("nan"))), 0.0005, rel_tol=0.0, abs_tol=1e-15
        )
        for row in isotropic_candidates
    ):
        raise ValueError("Radius artifact isotropic candidate metadata is invalid")
    recomputed_isotropic = dict(aggregate_candidate_metrics(isotropic_candidates))
    for key in ("symmetric_kl", "hidden_absolute_rms", "hidden_relative_rms"):
        if not math.isclose(
            float(recomputed_isotropic[key]),
            float(isotropic.get("aggregate", {}).get(key, float("nan"))),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Radius artifact isotropic aggregate {key} is invalid")

    recomputed_rows = []
    for radius_row in payload.get("modular_radii", []):
        radius = float(radius_row["radius"])
        candidates = radius_row.get("candidates", [])
        if [int(row.get("seed", -1)) for row in candidates] != candidate_pool[:prefix_size]:
            raise ValueError(
                f"Radius artifact candidates do not use the fixed prefix at r={radius}"
            )
        if any(
            row.get("method") != "recursive_modular_shell_v2"
            or not math.isclose(
                float(row.get("radius", float("nan"))),
                radius,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for row in candidates
        ):
            raise ValueError(f"Radius artifact candidate metadata is invalid at r={radius}")
        aggregate = dict(aggregate_candidate_metrics(candidates))
        for key in ("symmetric_kl", "hidden_absolute_rms", "hidden_relative_rms"):
            if not math.isclose(
                float(aggregate[key]),
                float(radius_row.get("aggregate", {}).get(key, float("nan"))),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"Radius artifact modular aggregate {key} is invalid at "
                    f"r={radius}"
                )
        objective_j = radius_objective(
            aggregate["symmetric_kl"],
            recomputed_isotropic["symmetric_kl"],
            aggregate["hidden_relative_rms"],
            recomputed_isotropic["hidden_relative_rms"],
        )
        if not math.isclose(
            objective_j,
            float(radius_row.get("objective_j", float("nan"))),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Radius artifact objective J is invalid at r={radius}"
            )
        recomputed_rows.append(
            {"radius": radius, "objective_j": objective_j}
        )
    if not recomputed_rows:
        raise ValueError("Radius artifact has no Modular-Shell radius rows")
    recomputed_selection = select_radius(recomputed_rows)
    if not math.isclose(
        float(recomputed_selection["radius"]),
        selected_radius,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Radius artifact selected_radius is not argmin J(r)")
    if not math.isclose(
        float(recomputed_selection["objective_j"]),
        float(payload.get("selected_objective_j", float("nan"))),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("Radius artifact selected_objective_j is invalid")
    return payload


def load_and_validate_distance_artifact(
    args: argparse.Namespace,
    train_path: str | Path,
    test_path: str | Path,
) -> Mapping[str, Any] | None:
    """Bind one run to an accuracy-blind functional-distance target."""

    if args.distance_match_artifact is None:
        return None
    payload, resolved_revision = validate_distance_match_artifact(
        args.distance_match_artifact,
        repository_root=REPO_ROOT,
        target_name=args.distance_match_target,
        dataset=args.dataset,
        model_name=args.model_name,
        model_revision=args.model_revision,
        perturbation_method=args.perturbation_method,
        radius=args.radius,
        mass_config=args.mass_config,
        sensitivity_profile=args.sensitivity_profile,
        population_size=args.population_size,
        population_prefixes=args.population_prefix_list,
        top_k_values=args.top_k_list,
        global_seed=args.global_seed,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        train_data_path=train_path,
        test_data_path=test_path,
    )
    if resolved_revision is not None:
        args.model_revision = resolved_revision
    return payload


def _append_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_metadata() -> Mapping[str, Any]:
    def command(*parts: str) -> str | None:
        try:
            return subprocess.check_output(
                parts,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = command("git", "status", "--porcelain=v1")
    diff = command("git", "diff", "--binary", "HEAD")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(
            (diff or "").encode("utf-8")
        ).hexdigest(),
    }


def _visible_gpu_count(cuda_devices: str) -> int:
    devices = [item.strip() for item in cuda_devices.split(",") if item.strip()]
    return max(1, len(devices))


def _synchronize_cuda_worker(_worker: Any) -> bool:
    """Synchronize the CUDA stream inside one vLLM worker process."""

    # Import inside the callable so Ray/cloudpickle does not capture PyTorch's
    # non-pickleable module-level CudnnModule object in the function closure.
    import torch as worker_torch

    if worker_torch.cuda.is_available():
        worker_torch.cuda.synchronize()
    return True


def _accelerator_metadata_worker(_worker: Any) -> Mapping[str, Any]:
    """Collect accelerator metadata inside one vLLM worker process."""

    # See _synchronize_cuda_worker: this callable crosses two Ray/vLLM
    # serialization boundaries and therefore must not close over global torch.
    import torch as worker_torch

    if not worker_torch.cuda.is_available():
        return {
            "available": False,
            "torch_cuda_version": worker_torch.version.cuda,
        }
    device_index = worker_torch.cuda.current_device()
    properties = worker_torch.cuda.get_device_properties(device_index)
    return {
        "available": True,
        "device_index": int(device_index),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "torch_cuda_version": worker_torch.version.cuda,
        "cudnn_version": worker_torch.backends.cudnn.version(),
    }


def _synchronize_cuda_engines(engines: Sequence[Any]) -> None:
    """Synchronize every tensor-parallel worker owned by every engine."""

    ray.get(
        [
            engine.collective_rpc.remote(_synchronize_cuda_worker, args=())
            for engine in engines
        ]
    )


def _flatten_accelerator_metadata(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, (list, tuple)):
        rows: List[Mapping[str, Any]] = []
        for item in payload:
            rows.extend(_flatten_accelerator_metadata(item))
        return rows
    return []


def _collect_accelerator_metadata(
    engines: Sequence[Any],
) -> List[Mapping[str, Any]]:
    raw = ray.get(
        [
            engine.collective_rpc.remote(_accelerator_metadata_worker, args=())
            for engine in engines
        ]
    )
    return _flatten_accelerator_metadata(raw)


def load_data(handler: Any, args: argparse.Namespace) -> Tuple[List[Any], List[Any]]:
    train_path = args.train_data_path or handler.default_train_path
    test_path = args.test_data_path or handler.default_test_path
    print(f"Loading {handler.name} data...")

    if train_path == test_path:
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        train_datas = all_data[: args.train_samples]
        stop = None if args.test_samples is None else args.train_samples + args.test_samples
        test_datas = all_data[args.train_samples : stop]
        if len(test_datas) < 50:
            raise ValueError(
                "The train/test paths are identical and leave fewer than 50 held-out "
                "examples. Provide a separate validation file for this experiment."
            )
    else:
        train_datas = handler.load_data(
            train_path, split="train", max_samples=args.train_samples
        )
        test_datas = handler.load_data(
            test_path, split="test", max_samples=args.test_samples
        )

    print(f"Train: {len(train_datas)} | Validation: {len(test_datas)}")
    return train_datas, test_datas


def format_prompts(
    tokenizer: AutoTokenizer,
    model_name: str,
    datas: Sequence[Mapping[str, Any]],
    chat_template_date: str = DEFAULT_CHAT_TEMPLATE_DATE,
) -> List[str]:
    return render_prompts(
        tokenizer,
        model_name,
        datas,
        chat_template_date=chat_template_date,
    )

def prepare_generation_prompts(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
) -> Tuple[List[TokensPrompt], List[List[int]]]:
    token_ids = encode_rendered_prompts(tokenizer, prompts)
    return [TokensPrompt(prompt_token_ids=ids) for ids in token_ids], token_ids


def count_prompt_tokens(token_ids: Sequence[Sequence[int]]) -> int:
    return sum(len(ids) for ids in token_ids)


def count_completion_tokens(outputs: Sequence[Any]) -> int:
    total = 0
    for output in outputs:
        if not output.outputs:
            continue
        token_ids = getattr(output.outputs[0], "token_ids", None)
        if token_ids is not None:
            total += len(token_ids)
    return total


def extract_answer(handler: Any, response_text: str, data: Mapping[str, Any]) -> str:
    if handler.name == "countdown":
        numbers = data.get("numbers")
        answer, is_valid, _ = handler.extract_answer_for_voting(
            response_text, numbers=numbers
        )
        return answer if is_valid else ""
    if hasattr(handler, "extract_answer_for_voting"):
        return handler.extract_answer_for_voting(response_text) or ""
    return handler.extract_answer(response_text) or ""


def answer_is_correct(handler: Any, answer: str, data: Mapping[str, Any]) -> bool:
    if not answer:
        return False
    if hasattr(handler, "is_voted_answer_correct"):
        return bool(handler.is_voted_answer_correct(answer, data["ground_truth"]))
    formatted = handler.format_answer_for_check(answer)
    return bool(handler.is_answer_correct(formatted, data["ground_truth"]))


def perturbation_rpc_args(args: argparse.Namespace, seed: int) -> Tuple[Any, ...]:
    return (
        int(seed),
        float(args.radius),
        False,
        args.perturbation_method,
        args.mass_config,
        int(args.power_iterations),
        args.sensitivity_profile,
    )


def evaluate_base_model(
    engines: Sequence[Any],
    handler: Any,
    train_prompts: Sequence[str | TokensPrompt],
    validation_prompts: Sequence[str | TokensPrompt],
    train_datas: Sequence[Mapping[str, Any]],
    validation_datas: Sequence[Mapping[str, Any]],
    sampling_params: SamplingParams,
    run_dir: Path,
    save_predictions: bool,
) -> Tuple[float, float, Dict[str, int], Dict[str, float]]:
    print(f"\n{'=' * 72}\nBASE MODEL EVALUATION\n{'=' * 72}")
    timing: Dict[str, float] = {}
    tokens = {"base_train_completion_tokens": 0, "base_validation_completion_tokens": 0}

    start = time.perf_counter()
    train_outputs = ray.get(
        engines[0].generate.remote(train_prompts, sampling_params, use_tqdm=False)
    )
    timing["base_train_generation_sec"] = time.perf_counter() - start
    tokens["base_train_completion_tokens"] = count_completion_tokens(train_outputs)
    base_train_reward = float(handler.postprocess_outputs(train_outputs, train_datas))
    print(f"Base train reward: {base_train_reward:.6f}")
    del train_outputs
    gc.collect()

    start = time.perf_counter()
    validation_outputs = ray.get(
        engines[0].generate.remote(validation_prompts, sampling_params, use_tqdm=False)
    )
    timing["base_validation_generation_sec"] = time.perf_counter() - start
    tokens["base_validation_completion_tokens"] = count_completion_tokens(
        validation_outputs
    )

    correct = 0
    rows = []
    for index, (output, data) in enumerate(zip(validation_outputs, validation_datas)):
        response_text = output.outputs[0].text if output.outputs else ""
        answer = extract_answer(handler, response_text, data)
        is_correct = answer_is_correct(handler, answer, data)
        correct += int(is_correct)
        if save_predictions:
            rows.append(
                {
                    "sample_index": index,
                    "answer": answer,
                    "correct": is_correct,
                }
            )

    base_validation_accuracy = correct / len(validation_datas) if validation_datas else 0.0
    print(
        f"Base validation accuracy: {base_validation_accuracy * 100:.2f}% "
        f"({correct}/{len(validation_datas)})"
    )
    if save_predictions:
        _append_jsonl(run_dir / "base_predictions.jsonl", rows)

    del validation_outputs
    gc.collect()
    timing["base_evaluation_sec"] = sum(timing.values())
    return base_train_reward, base_validation_accuracy, tokens, timing


def run_sampling(
    args: argparse.Namespace,
    engines: Sequence[Any],
    handler: Any,
    train_prompts: Sequence[str | TokensPrompt],
    train_datas: Sequence[Mapping[str, Any]],
    sampling_params: SamplingParams,
    run_dir: Path,
) -> Tuple[List[CandidateRecord], Dict[str, float], int]:
    print(f"\n{'=' * 72}\nPERTURBATION SAMPLING\n{'=' * 72}")
    print(
        f"Method={args.perturbation_method} | Radius={args.radius} | "
        f"Population={args.population_size}"
    )

    if args.wall_clock_mode:
        # Preserve the exact candidate identities used by the existing
        # Nmax=300 population-scaling run while evaluating only the requested
        # N=25/100 candidates.  Constructing integer seeds is not candidate
        # generation; no omitted candidate is perturbed or evaluated.
        all_seeds = build_candidate_seeds(args.global_seed, 300)[
            : args.population_size
        ]
    else:
        rng = np.random.default_rng(seed=args.global_seed)
        all_seeds = rng.choice(
            2**31, size=args.population_size, replace=False
        ).astype(np.int64).tolist()

    timings = {
        "perturbation_apply_sec": 0.0,
        "candidate_generation_sec": 0.0,
        "perturbation_restore_sec": 0.0,
        "reward_postprocess_sec": 0.0,
    }
    completion_tokens = 0
    records: List[CandidateRecord] = []
    records_path = run_dir / "candidate_records.jsonl"

    candidate_index = 0
    batch_index = 0
    while candidate_index < args.population_size:
        batch_size = min(args.num_engines, args.population_size - candidate_index)
        batch_seeds = all_seeds[candidate_index : candidate_index + batch_size]

        start = time.perf_counter()
        ray.get(
            [
                engines[engine_index].collective_rpc.remote(
                    "perturb_self_weights",
                    args=perturbation_rpc_args(args, seed),
                )
                for engine_index, seed in enumerate(batch_seeds)
            ]
        )
        timings["perturbation_apply_sec"] += time.perf_counter() - start

        start = time.perf_counter()
        batch_outputs = ray.get(
            [
                engines[engine_index].generate.remote(
                    train_prompts, sampling_params, use_tqdm=False
                )
                for engine_index in range(batch_size)
            ]
        )
        timings["candidate_generation_sec"] += time.perf_counter() - start
        for outputs in batch_outputs:
            completion_tokens += count_completion_tokens(outputs)

        start = time.perf_counter()
        ray.get(
            [
                engines[engine_index].collective_rpc.remote(
                    "restore_self_weights",
                    args=perturbation_rpc_args(args, seed),
                )
                for engine_index, seed in enumerate(batch_seeds)
            ]
        )
        timings["perturbation_restore_sec"] += time.perf_counter() - start

        start = time.perf_counter()
        new_records: List[CandidateRecord] = []
        for local_index, seed in enumerate(batch_seeds):
            reward = float(
                handler.postprocess_outputs(batch_outputs[local_index], train_datas)
            )
            record = {
                "candidate_index": candidate_index + local_index,
                "seed": int(seed),
                "radius": float(args.radius),
                "train_reward": reward,
                "batch_index": batch_index,
            }
            records.append(record)
            new_records.append(record)
        timings["reward_postprocess_sec"] += time.perf_counter() - start
        _append_jsonl(records_path, new_records)

        candidate_index += batch_size
        batch_index += 1
        reward_text = ", ".join(f"{row['train_reward']:.4f}" for row in new_records)
        print(
            f"Batch {batch_index} | {candidate_index}/{args.population_size} | "
            f"rewards=[{reward_text}]"
        )

        del batch_outputs
        gc.collect()

    timings["sampling_total_sec"] = sum(
        value for key, value in timings.items() if key != "sampling_total_sec"
    )
    return records, timings, completion_tokens


def rank_prefixes(
    records: Sequence[CandidateRecord],
    prefixes: Sequence[int],
    top_k_values: Sequence[int],
) -> Tuple[Dict[int, List[CandidateRecord]], List[CandidateRecord]]:
    max_k = max(top_k_values)
    selections: Dict[int, List[CandidateRecord]] = {}
    union_by_seed: Dict[int, CandidateRecord] = {}

    for prefix in prefixes:
        ranked = sorted(
            records[:prefix],
            key=lambda row: (-float(row["train_reward"]), int(row["candidate_index"])),
        )
        selected = [dict(row) for row in ranked[: min(max_k, prefix)]]
        for rank, row in enumerate(selected, start=1):
            row["rank_within_prefix"] = rank
            union_by_seed.setdefault(int(row["seed"]), row)
        selections[prefix] = selected

    union = sorted(
        union_by_seed.values(), key=lambda row: int(row["candidate_index"])
    )
    return selections, union


def evaluate_selected_experts(
    args: argparse.Namespace,
    engines: Sequence[Any],
    handler: Any,
    validation_prompts: Sequence[str],
    validation_datas: Sequence[Mapping[str, Any]],
    sampling_params: SamplingParams,
    selected_union: Sequence[CandidateRecord],
    run_dir: Path,
    save_predictions: bool,
) -> Tuple[Dict[int, List[str]], Dict[str, float], int]:
    print(f"\n{'=' * 72}\nUNION EXPERT EVALUATION\n{'=' * 72}")
    print(
        f"Unique experts required by all prefixes: {len(selected_union)} "
        f"(max K={max(args.top_k_list)})"
    )

    timings = {
        "expert_perturbation_apply_sec": 0.0,
        "expert_validation_generation_sec": 0.0,
        "expert_perturbation_restore_sec": 0.0,
        "expert_answer_extraction_sec": 0.0,
    }
    completion_tokens = 0
    answers_by_seed: Dict[int, List[str]] = {}
    prediction_path = run_dir / "expert_predictions.jsonl"

    cursor = 0
    batch_index = 0
    while cursor < len(selected_union):
        batch = selected_union[cursor : cursor + args.num_engines]

        start = time.perf_counter()
        ray.get(
            [
                engines[engine_index].collective_rpc.remote(
                    "perturb_self_weights",
                    args=perturbation_rpc_args(args, int(record["seed"])),
                )
                for engine_index, record in enumerate(batch)
            ]
        )
        timings["expert_perturbation_apply_sec"] += time.perf_counter() - start

        start = time.perf_counter()
        batch_outputs = ray.get(
            [
                engines[engine_index].generate.remote(
                    validation_prompts, sampling_params, use_tqdm=False
                )
                for engine_index in range(len(batch))
            ]
        )
        timings["expert_validation_generation_sec"] += time.perf_counter() - start
        for outputs in batch_outputs:
            completion_tokens += count_completion_tokens(outputs)

        start = time.perf_counter()
        ray.get(
            [
                engines[engine_index].collective_rpc.remote(
                    "restore_self_weights",
                    args=perturbation_rpc_args(args, int(record["seed"])),
                )
                for engine_index, record in enumerate(batch)
            ]
        )
        timings["expert_perturbation_restore_sec"] += time.perf_counter() - start

        start = time.perf_counter()
        prediction_rows: List[Mapping[str, Any]] = []
        for local_index, record in enumerate(batch):
            seed = int(record["seed"])
            answers: List[str] = []
            correctness: List[bool] = []
            for output, data in zip(batch_outputs[local_index], validation_datas):
                response_text = output.outputs[0].text if output.outputs else ""
                answer = extract_answer(handler, response_text, data)
                answers.append(answer)
                correctness.append(answer_is_correct(handler, answer, data))
            answers_by_seed[seed] = answers
            if save_predictions:
                prediction_rows.append(
                    {
                        "candidate_index": int(record["candidate_index"]),
                        "seed": seed,
                        "radius": float(record["radius"]),
                        "train_reward": float(record["train_reward"]),
                        "answers": answers,
                        "correct": correctness,
                    }
                )
        timings["expert_answer_extraction_sec"] += time.perf_counter() - start
        if save_predictions:
            _append_jsonl(prediction_path, prediction_rows)

        cursor += len(batch)
        batch_index += 1
        print(
            f"Expert batch {batch_index} | {cursor}/{len(selected_union)} completed"
        )
        del batch_outputs
        gc.collect()

    timings["expert_evaluation_total_sec"] = sum(
        value for key, value in timings.items() if key != "expert_evaluation_total_sec"
    )
    return answers_by_seed, timings, completion_tokens


def candidate_summary(
    records: Sequence[CandidateRecord], base_train_reward: float | None
) -> Dict[str, float | None]:
    rewards = np.asarray([float(row["train_reward"]) for row in records], dtype=np.float64)
    ranked = np.sort(rewards)[::-1]

    def top_mean(k: int) -> float:
        return float(np.mean(ranked[: min(k, len(ranked))]))

    return {
        "mean_train_reward": float(np.mean(rewards)),
        "std_train_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "solution_density": (
            None
            if base_train_reward is None
            else float(np.mean(rewards > base_train_reward))
        ),
        "top1_train_reward": float(ranked[0]),
        "top5_mean_train_reward": top_mean(5),
        "top10_mean_train_reward": top_mean(10),
        "p90_train_reward": float(np.quantile(rewards, 0.90)),
        "p95_train_reward": float(np.quantile(rewards, 0.95)),
        "p99_train_reward": float(np.quantile(rewards, 0.99)),
    }


def evaluate_prefix_ensembles(
    args: argparse.Namespace,
    handler: Any,
    validation_datas: Sequence[Mapping[str, Any]],
    selections: Mapping[int, Sequence[CandidateRecord]],
    answers_by_seed: Mapping[int, Sequence[str]],
    base_validation_accuracy: float | None,
    run_dir: Path,
    save_predictions: bool,
    all_records: Sequence[CandidateRecord],
    base_train_reward: float | None,
) -> Tuple[Dict[str, Any], float]:
    print(f"\n{'=' * 72}\nPREFIX ENSEMBLE EVALUATION\n{'=' * 72}")
    start = time.perf_counter()
    results: Dict[str, Any] = {}
    prediction_rows: List[Mapping[str, Any]] = []

    for prefix in args.population_prefix_list:
        selected = list(selections[prefix])
        prefix_result: Dict[str, Any] = {
            "candidate_statistics": candidate_summary(
                all_records[:prefix], base_train_reward
            ),
            "ensemble_results": {},
        }

        for k_value in args.top_k_list:
            experts = selected[:k_value]
            correct = 0
            valid_vote_counts: List[int] = []
            selected_rewards = [float(row["train_reward"]) for row in experts]

            for sample_index, data in enumerate(validation_datas):
                answers = [
                    answers_by_seed[int(expert["seed"])][sample_index]
                    for expert in experts
                    if answers_by_seed[int(expert["seed"])][sample_index]
                ]
                valid_vote_counts.append(len(answers))
                voted_answer = Counter(answers).most_common(1)[0][0] if answers else ""
                is_correct = answer_is_correct(handler, voted_answer, data)
                correct += int(is_correct)

                if save_predictions:
                    prediction_rows.append(
                        {
                            "population_size": prefix,
                            "k": k_value,
                            "sample_index": sample_index,
                            "voted_answer": voted_answer,
                            "correct": is_correct,
                            "num_valid_votes": len(answers),
                        }
                    )

            accuracy = correct / len(validation_datas) if validation_datas else 0.0
            result = {
                "accuracy": accuracy,
                "correct": correct,
                "num_samples": len(validation_datas),
                "gain_over_base": (
                    None
                    if base_validation_accuracy is None
                    else accuracy - base_validation_accuracy
                ),
                "mean_selected_train_reward": float(np.mean(selected_rewards)),
                "mean_valid_votes": float(np.mean(valid_vote_counts)),
                "selected_candidate_indices": [
                    int(row["candidate_index"]) for row in experts
                ],
                "selected_seeds": [int(row["seed"]) for row in experts],
            }
            prefix_result["ensemble_results"][str(k_value)] = result
            gain_text = (
                ""
                if base_validation_accuracy is None
                else (
                    f", gain={100 * (accuracy - base_validation_accuracy):+.2f} pt"
                )
            )
            print(
                f"N={prefix:>3}, K={k_value:>2}: {accuracy * 100:.2f}% "
                f"({correct}/{len(validation_datas)}){gain_text}"
            )

        results[str(prefix)] = prefix_result

    if save_predictions:
        _append_jsonl(run_dir / "ensemble_predictions.jsonl", prediction_rows)
    return results, time.perf_counter() - start


def serialize_selections(
    selections: Mapping[int, Sequence[CandidateRecord]],
) -> Dict[str, List[Mapping[str, Any]]]:
    serialized: Dict[str, List[Mapping[str, Any]]] = {}
    for prefix, rows in selections.items():
        serialized[str(prefix)] = [
            {
                "rank": rank,
                "candidate_index": int(row["candidate_index"]),
                "seed": int(row["seed"]),
                "radius": float(row["radius"]),
                "train_reward": float(row["train_reward"]),
            }
            for rank, row in enumerate(rows, start=1)
        ]
    return serialized




def _unwrap_collective_rpc_payload(payload: Any) -> Any:
    """Unwrap common vLLM collective_rpc return containers."""
    current = payload
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    return current


def save_recursive_modular_diagnostics(
    engine: Any,
    method: str,
    mass_config: Mapping[str, float],
    sensitivity_profile: Mapping[str, Any] | None,
    power_iterations: int,
    run_dir: Path,
) -> Dict[str, Any]:
    """Build and persist the recursive Qwen shadow-tree on the actual worker."""
    raw = ray.get(
        engine.collective_rpc.remote(
            "get_recursive_modular_diagnostics",
            args=(
                mass_config,
                method,
                sensitivity_profile,
                int(power_iterations),
            ),
        )
    )
    diagnostics = _unwrap_collective_rpc_payload(raw)
    if not isinstance(diagnostics, Mapping):
        raise TypeError(
            "get_recursive_modular_diagnostics returned a non-mapping payload: "
            f"{type(diagnostics)!r}"
        )
    diagnostics = dict(diagnostics)
    _json_dump(run_dir / "recursive_module_tree.json", diagnostics.get("tree", {}))
    _json_dump(
        run_dir / "recursive_scale_map.json",
        {
            row["parameter_name"]: row["scale"]
            for row in diagnostics.get("parameters", [])
        },
    )
    _json_dump(
        run_dir / "recursive_scale_statistics.json",
        {
            key: value
            for key, value in diagnostics.items()
            if key not in {"tree", "parameters"}
        },
    )
    _json_dump(run_dir / "recursive_parameter_scales.json", diagnostics.get("parameters", []))
    return diagnostics


def main(args: argparse.Namespace) -> None:
    total_start = time.perf_counter()
    train_path = args.train_data_path or get_dataset_handler(args.dataset).default_train_path
    test_path = args.test_data_path or get_dataset_handler(args.dataset).default_test_path
    radius_artifact = load_and_validate_radius_artifact(args)
    distance_artifact = load_and_validate_distance_artifact(
        args, train_path, test_path
    )
    distance_target = (
        None
        if distance_artifact is None
        else next(
            row
            for row in distance_artifact["targets"]
            if row["name"] == args.distance_match_target
        )
    )
    run_source_manifest = source_manifest(REPO_ROOT)
    run_dir = create_run_dir(args)
    start_console_tee(str(run_dir))

    environment = {
        "git_commit": _git_commit(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "ray": _package_version("ray"),
        "vllm": _package_version("vllm"),
        "transformers": _package_version("transformers"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": args.cuda_devices,
        "num_visible_gpus": _visible_gpu_count(args.cuda_devices),
        "num_engines": args.num_engines,
        "tensor_parallel_size": args.tp,
        "dtype": args.precision,
        "inference_backend": "vllm",
        "wall_clock_protocol": (
            WALL_CLOCK_PROTOCOL if args.wall_clock_mode else None
        ),
        "execution_protocol": "official_randopt_j_v1",
        "model_revision": args.model_revision,
        "git": _git_metadata(),
        "source_manifest": run_source_manifest,
        "train_data_sha256": _path_fingerprint(train_path),
        "test_data_sha256": _path_fingerprint(test_path),
    }
    if args.wall_clock_mode:
        wall_clock_source_paths = (
            "population_scaling.py",
            "utils/wall_clock.py",
            "scripts/wall_clock/run_wall_clock_k25.sh",
            "scripts/wall_clock/summarize_wall_clock.py",
        )
        environment["wall_clock_source_sha256"] = {
            path: _sha256_file(REPO_ROOT / path)
            for path in wall_clock_source_paths
        }
    _json_dump(run_dir / "environment.json", environment)
    _json_dump(run_dir / "args.json", vars(args))
    if args.sensitivity_profile is not None:
        _json_dump(run_dir / "sensitivity_profile.json", args.sensitivity_profile)
    if radius_artifact is not None:
        _json_dump(run_dir / "radius_selection_artifact.json", radius_artifact)
    if distance_artifact is not None:
        _json_dump(run_dir / "distance_match_artifact.json", distance_artifact)
    _json_dump(run_dir / "source_manifest.json", run_source_manifest)

    print(f"Run directory: {run_dir}")
    print(f"Model: {args.model_name}")
    print(
        f"Method: {args.perturbation_method} | Radius: {args.radius} | "
        f"Seed: {args.global_seed}"
    )
    print(
        f"Population prefixes: {args.population_prefix_list} | "
        f"K values: {args.top_k_list}"
    )

    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens
    timings: Dict[str, float] = {}
    token_counts: Dict[str, int] = {}
    engines: Sequence[Any] = []
    pgs: Sequence[Any] = []

    setup_start = time.perf_counter()
    train_datas, validation_datas = load_data(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    environment["resolved_model_revision"] = getattr(
        tokenizer, "init_kwargs", {}
    ).get("_commit_hash")
    train_rendered_prompts = format_prompts(
        tokenizer,
        args.model_name,
        train_datas,
        chat_template_date=args.chat_template_date,
    )
    validation_rendered_prompts = format_prompts(
        tokenizer,
        args.model_name,
        validation_datas,
        chat_template_date=args.chat_template_date,
    )
    train_prompts, train_prompt_token_ids = prepare_generation_prompts(
        tokenizer, train_rendered_prompts
    )
    validation_prompts, validation_prompt_token_ids = prepare_generation_prompts(
        tokenizer, validation_rendered_prompts
    )
    train_prompt_manifest = prompt_manifest(
        tokenizer,
        train_prompt_token_ids,
        chat_template_date=args.chat_template_date,
    )
    validation_prompt_manifest = prompt_manifest(
        tokenizer,
        validation_prompt_token_ids,
        chat_template_date=args.chat_template_date,
    )
    _json_dump(
        run_dir / "prompt_tokenization.json",
        {"train": train_prompt_manifest, "validation": validation_prompt_manifest},
    )
    train_prompt_tokens_per_model = count_prompt_tokens(train_prompt_token_ids)
    validation_prompt_tokens_per_model = count_prompt_tokens(
        validation_prompt_token_ids
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=args.global_seed,
        max_tokens=max_tokens,
    )

    try:
        if os.environ.get("RAY_ADDRESS"):
            ray.init(address="auto", ignore_reinit_error=True)
        else:
            ray.init(address="local", ignore_reinit_error=True)

        engines, pgs = launch_engines(
            args.num_engines,
            args.model_name,
            precision=args.precision,
            tensor_parallel_size=args.tp,
            revision=args.model_revision,
        )
        timings["setup_and_engine_launch_sec"] = time.perf_counter() - setup_start
        if args.wall_clock_mode:
            accelerator_metadata = _collect_accelerator_metadata(engines)
            environment["accelerators"] = accelerator_metadata
            environment["gpu_names"] = [
                row.get("name")
                for row in accelerator_metadata
                if row.get("available") and row.get("name")
            ]
            _json_dump(run_dir / "environment.json", environment)
    except BaseException:
        # Setup failures used to occur before the main finally block, leaving
        # detached placement groups or GPU workers behind. Always tear down the
        # local Ray runtime before propagating a launch/metadata error.
        if engines:
            cleanup_engines(engines, pgs)
        else:
            ray.shutdown()
        stop_console_tee()
        raise

    recursive_diagnostics: Dict[str, Any] | None = None
    wall_search_start: float | None = None
    wall_ensemble_start: float | None = None

    try:
        if not args.wall_clock_mode and args.perturbation_method in {
            "recursive_modular_shell",
            "recursive_modular_shell_v2",
        }:
            recursive_diagnostics = save_recursive_modular_diagnostics(
                engines[0],
                args.perturbation_method,
                args.mass_config,
                args.sensitivity_profile,
                args.power_iterations,
                run_dir,
            )
            print(
                f"Recursive modular diagnostics saved (version={recursive_diagnostics.get('version')}): "
                f"layers={recursive_diagnostics.get('num_layers_detected')}, "
                f"active_parameters={recursive_diagnostics.get('num_active_parameters')}"
            )

        base_train_reward: float | None
        base_validation_accuracy: float | None
        if args.wall_clock_mode:
            # Base-model accuracy is not part of the requested search or
            # ensemble endpoint. Omitting this auxiliary full-dataset pass
            # keeps setup aligned with model/data/tokenizer initialization.
            base_train_reward = None
            base_validation_accuracy = None
            token_counts.update(
                {
                    "base_train_completion_tokens": 0,
                    "base_validation_completion_tokens": 0,
                }
            )
        else:
            base_train_reward, base_validation_accuracy, base_tokens, base_timing = (
                evaluate_base_model(
                    engines=engines,
                    handler=handler,
                    train_prompts=train_prompts,
                    validation_prompts=validation_prompts,
                    train_datas=train_datas,
                    validation_datas=validation_datas,
                    sampling_params=sampling_params,
                    run_dir=run_dir,
                    save_predictions=not args.skip_prediction_files,
                )
            )
            timings.update(base_timing)
            token_counts.update(base_tokens)

        if args.base_only:
            token_counts.update(
                {
                    "base_train_prompt_tokens": train_prompt_tokens_per_model,
                    "base_validation_prompt_tokens": validation_prompt_tokens_per_model,
                }
            )
            timings["total_wall_time_sec"] = time.perf_counter() - total_start
            timings["gpu_hours"] = (
                timings["total_wall_time_sec"]
                * _visible_gpu_count(args.cuda_devices)
                / 3600.0
            )
            base_only_results = {
                "status": "complete",
                "base_only": True,
                "dataset": args.dataset,
                "model": args.model_name,
                "model_revision": args.model_revision,
                "global_seed": args.global_seed,
                "base_train_reward": base_train_reward,
                "base_validation_accuracy": base_validation_accuracy,
                "train_samples": len(train_datas),
                "validation_samples": len(validation_datas),
                "chat_template_date": args.chat_template_date,
                "prompt_tokenization": {
                    "train": train_prompt_manifest,
                    "validation": validation_prompt_manifest,
                },
                "timing": timings,
                "token_counts": token_counts,
                "environment": environment,
            }
            _json_dump(run_dir / "base_only_results.json", base_only_results)
            _json_dump(run_dir / "timing.json", timings)
            _json_dump(run_dir / "token_counts.json", token_counts)
            print(f"\n{'=' * 72}\nBASE-ONLY COMPLETE\n{'=' * 72}")
            print(f"Results: {run_dir / 'base_only_results.json'}")
            print(f"Wall time: {timings['total_wall_time_sec'] / 3600:.3f} hours")
            return

        if args.wall_clock_mode:
            # Setup ends after model/data/tokenizer initialization. The
            # method-specific recursive scale construction remains inside the
            # search phase.
            _synchronize_cuda_engines(engines)
            wall_search_start = time.perf_counter()
            timings["setup_sec"] = wall_search_start - total_start

        if args.wall_clock_mode and args.perturbation_method in {
            "recursive_modular_shell",
            "recursive_modular_shell_v2",
        }:
            recursive_diagnostics = save_recursive_modular_diagnostics(
                engines[0],
                args.perturbation_method,
                args.mass_config,
                args.sensitivity_profile,
                args.power_iterations,
                run_dir,
            )
            print(
                f"Recursive modular diagnostics saved (version={recursive_diagnostics.get('version')}): "
                f"layers={recursive_diagnostics.get('num_layers_detected')}, "
                f"active_parameters={recursive_diagnostics.get('num_active_parameters')}"
            )

        candidate_records, sampling_timing, sampling_completion_tokens = run_sampling(
            args=args,
            engines=engines,
            handler=handler,
            train_prompts=train_prompts,
            train_datas=train_datas,
            sampling_params=sampling_params,
            run_dir=run_dir,
        )
        timings.update(sampling_timing)
        token_counts["candidate_train_completion_tokens"] = sampling_completion_tokens

        selection_start = time.perf_counter()
        selections, selected_union = rank_prefixes(
            records=candidate_records,
            prefixes=args.population_prefix_list,
            top_k_values=args.top_k_list,
        )
        timings["selection_sec"] = time.perf_counter() - selection_start
        _json_dump(
            run_dir / "selected_models_by_prefix.json",
            serialize_selections(selections),
        )

        if args.wall_clock_mode:
            assert wall_search_start is not None
            _synchronize_cuda_engines(engines)
            wall_ensemble_start = time.perf_counter()
            timings["search_sec"] = wall_ensemble_start - wall_search_start

        answers_by_seed, expert_timing, expert_completion_tokens = (
            evaluate_selected_experts(
                args=args,
                engines=engines,
                handler=handler,
                validation_prompts=validation_prompts,
                validation_datas=validation_datas,
                sampling_params=sampling_params,
                selected_union=selected_union,
                run_dir=run_dir,
                save_predictions=not args.skip_prediction_files,
            )
        )
        timings.update(expert_timing)
        token_counts["selected_expert_validation_completion_tokens"] = (
            expert_completion_tokens
        )

        population_results, voting_sec = evaluate_prefix_ensembles(
            args=args,
            handler=handler,
            validation_datas=validation_datas,
            selections=selections,
            answers_by_seed=answers_by_seed,
            base_validation_accuracy=base_validation_accuracy,
            run_dir=run_dir,
            save_predictions=not args.skip_prediction_files,
            all_records=candidate_records,
            base_train_reward=base_train_reward,
        )
        timings["prefix_voting_sec"] = voting_sec

        token_counts.update(
            {
                "base_train_prompt_tokens": (
                    0 if args.wall_clock_mode else train_prompt_tokens_per_model
                ),
                "base_validation_prompt_tokens": (
                    0 if args.wall_clock_mode else validation_prompt_tokens_per_model
                ),
                "candidate_train_prompt_tokens": (
                    train_prompt_tokens_per_model * args.population_size
                ),
                "selected_expert_validation_prompt_tokens": (
                    validation_prompt_tokens_per_model * len(selected_union)
                ),
            }
        )
        token_counts["total_prompt_tokens_estimated"] = sum(
            value for key, value in token_counts.items() if "prompt_tokens" in key
        )
        token_counts["total_completion_tokens"] = sum(
            value for key, value in token_counts.items() if "completion_tokens" in key
        )
        token_counts["total_tokens_estimated"] = (
            token_counts["total_prompt_tokens_estimated"]
            + token_counts["total_completion_tokens"]
        )

        if args.wall_clock_mode:
            assert wall_ensemble_start is not None
            _synchronize_cuda_engines(engines)
            wall_end = time.perf_counter()
            timings["ensemble_sec"] = wall_end - wall_ensemble_start
            timings["total_sec"] = wall_end - total_start
            timings["phase_sum_sec"] = (
                timings["setup_sec"]
                + timings["search_sec"]
                + timings["ensemble_sec"]
            )
            timings["phase_sum_residual_sec"] = (
                timings["total_sec"] - timings["phase_sum_sec"]
            )
            timings["total_wall_time_sec"] = timings["total_sec"]
        else:
            timings["total_wall_time_sec"] = time.perf_counter() - total_start
        timings["gpu_hours"] = (
            timings["total_wall_time_sec"]
            * _visible_gpu_count(args.cuda_devices)
            / 3600.0
        )
        timings["perturbation_apply_restore_sec"] = (
            timings.get("perturbation_apply_sec", 0.0)
            + timings.get("perturbation_restore_sec", 0.0)
            + timings.get("expert_perturbation_apply_sec", 0.0)
            + timings.get("expert_perturbation_restore_sec", 0.0)
        )

        wall_clock_record: Dict[str, Any] | None = None
        if args.wall_clock_mode:
            top_k = args.top_k_list[0]
            ensemble_accuracy = float(
                population_results[str(args.population_size)]["ensemble_results"]
                [str(top_k)]["accuracy"]
            )
            wall_clock_record = {
                "protocol": WALL_CLOCK_PROTOCOL,
                "method": wall_clock_method(args.perturbation_method),
                "perturbation_method": args.perturbation_method,
                "radius": float(args.radius),
                "mass_config": args.mass_config,
                "sensitivity_profile_fingerprint": sensitivity_profile_fingerprint(
                    args.sensitivity_profile
                ),
                "power_iterations": args.power_iterations,
                "model": args.model_name,
                "model_revision": args.model_revision,
                "resolved_model_revision": environment.get(
                    "resolved_model_revision"
                ),
                "task": args.dataset,
                "seed": args.global_seed,
                "population_size": args.population_size,
                "candidate_seed_pool_size": 300,
                "evaluated_candidate_count": args.population_size,
                "top_k": top_k,
                "train_samples": len(train_datas),
                "validation_samples": len(validation_datas),
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "chat_template_date": args.chat_template_date,
                "precision": args.precision,
                "setup_sec": timings["setup_sec"],
                "search_sec": timings["search_sec"],
                "ensemble_sec": timings["ensemble_sec"],
                "total_sec": timings["total_sec"],
                "phase_sum_residual_sec": timings["phase_sum_residual_sec"],
                "ensemble_accuracy": ensemble_accuracy,
                "gpu_synchronized": True,
                "gpu_synchronization": (
                    "vllm-collective-rpc-torch.cuda.synchronize"
                ),
                "setup_includes_base_evaluation": False,
                "search_includes_recursive_scale_construction": (
                    args.perturbation_method == "recursive_modular_shell_v2"
                ),
                "os_wall_time_path": (
                    None
                    if args.os_wall_time_path is None
                    else str(Path(args.os_wall_time_path).expanduser().resolve())
                ),
                "environment": environment,
            }
            _json_dump(run_dir / "wall_clock.json", wall_clock_record)

        results = {
            "status": "complete",
            "dataset": args.dataset,
            "model": args.model_name,
            "perturbation_method": args.perturbation_method,
            "radius": float(args.radius),
            "mass_config": args.mass_config,
            "sensitivity_profile_fingerprint": sensitivity_profile_fingerprint(
                args.sensitivity_profile
            ),
            "power_iterations": args.power_iterations,
            "execution_protocol": "official_randopt_j_v1",
            "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
            "noise_scheme": PARAMETER_NOISE_SCHEME,
            "weight_restore_scheme": WEIGHT_RESTORE_SCHEME,
            "per_candidate_exact_reset": False,
            "chat_template_date": args.chat_template_date,
            "prompt_tokenization": {
                "train": train_prompt_manifest,
                "validation": validation_prompt_manifest,
            },
            "radius_selection_artifact_sha256": (
                None
                if args.radius_selection_artifact is None
                else _sha256_file(args.radius_selection_artifact)
            ),
            "distance_match": (
                None
                if args.distance_match_artifact is None
                else {
                    "artifact_sha256": _sha256_file(args.distance_match_artifact),
                    "target": args.distance_match_target,
                    "target_median_output_kl": float(
                        distance_target["target_median_output_kl"]
                    ),
                    "method_scale": float(args.radius),
                    "selection_uses_accuracy": False,
                }
            ),
            "global_seed": args.global_seed,
            "train_samples": len(train_datas),
            "validation_samples": len(validation_datas),
            "population_size": args.population_size,
            "candidate_seed_pool_size": (
                300 if args.wall_clock_mode else args.population_size
            ),
            "population_prefixes": args.population_prefix_list,
            "top_k_values": args.top_k_list,
            "base_train_reward": base_train_reward,
            "base_validation_accuracy": base_validation_accuracy,
            "unique_validation_experts": len(selected_union),
            "population_results": population_results,
            "wall_clock": wall_clock_record,
            "timing": timings,
            "token_counts": token_counts,
            "environment": environment,
            "recursive_modular_summary": (
                None
                if recursive_diagnostics is None
                else {
                    key: value
                    for key, value in recursive_diagnostics.items()
                    if key not in {"tree", "parameters"}
                }
            ),
        }
        _json_dump(run_dir / "results.json", results)
        _json_dump(run_dir / "timing.json", timings)
        _json_dump(run_dir / "token_counts.json", token_counts)

        print(f"\n{'=' * 72}\nRUN COMPLETE\n{'=' * 72}")
        print(f"Results: {run_dir / 'results.json'}")
        print(f"Wall time: {timings['total_wall_time_sec'] / 3600:.3f} hours")
        print(f"GPU-hours: {timings['gpu_hours']:.3f}")
        print(f"Unique validation experts: {len(selected_union)}")
        if wall_clock_record is not None:
            print(
                "Wall-clock phases: "
                f"setup={wall_clock_record['setup_sec']:.2f}s, "
                f"search={wall_clock_record['search_sec']:.2f}s, "
                f"ensemble={wall_clock_record['ensemble_sec']:.2f}s, "
                f"total={wall_clock_record['total_sec']:.2f}s"
            )
            print(f"Wall-clock record: {run_dir / 'wall_clock.json'}")

    finally:
        cleanup_start = time.perf_counter()
        if engines:
            cleanup_engines(engines, pgs)
        cleanup_sec = time.perf_counter() - cleanup_start
        print(f"Cleanup time: {cleanup_sec:.2f} sec")
        stop_console_tee()


if __name__ == "__main__":
    main(parse_args())
