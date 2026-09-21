#!/usr/bin/env python3
"""Select a model-transfer Modular-Shell radius without task accuracy.

Noise generation, candidate seeds, and per-candidate subtraction restore match
the official RandOpt population search.  The only new operation is selecting
the Modular-Shell radius by functional displacement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ray  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from core import cleanup_engines, launch_engines  # noqa: E402
from data_handlers import get_dataset_handler, list_datasets  # noqa: E402
from utils.architecture_adapters import get_architecture_adapter  # noqa: E402
from utils.model_config import effective_text_config, transformers_causal_model_class  # noqa: E402
from utils.official_randopt_protocol import (  # noqa: E402
    CANDIDATE_SEED_SCHEME,
    PARAMETER_NOISE_SCHEME,
    WEIGHT_RESTORE_SCHEME,
    build_candidate_seeds,
)
from utils.official_randopt_provenance import source_manifest  # noqa: E402
from utils.official_prompt_protocol import (  # noqa: E402
    DEFAULT_CHAT_TEMPLATE_DATE,
    PROMPT_TOKENIZATION_SCHEME,
    encode_rendered_prompts,
    prompt_manifest as build_prompt_manifest,
    render_prompts,
)
from utils.functional_displacement import (  # noqa: E402
    aggregate_candidate_metrics,
    radius_objective,
    select_radius,
    symmetric_kl_from_logits,
)
from utils.hf_vllm_parameter_bridge import (  # noqa: E402
    apply_physical_perturbation,
    binding_manifest,
    build_physical_parameter_bindings,
    initialize_condition_from_base,
    parameter_drift_summary,
    snapshot_parameters,
)
from utils.perturbation_norms import (  # noqa: E402
    load_mass_config,
    load_sensitivity_profile,
    sensitivity_profile_fingerprint,
)


DEFAULT_RADII = "0.01,0.02,0.04,0.08,0.16,0.32,0.64"
def _float_csv(value: str) -> list[float]:
    try:
        values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("radii must be comma-separated floats") from exc
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("radii must contain positive values")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Accuracy-free functional displacement radius selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--model_revision",
        default=None,
        help="Optional immutable Hugging Face revision; resolved and saved when omitted",
    )
    parser.add_argument("--dataset", default="countdown", choices=list_datasets())
    parser.add_argument("--train_data_path", default=None)
    parser.add_argument(
        "--chat_template_date", default=DEFAULT_CHAT_TEMPLATE_DATE
    )
    parser.add_argument("--sensitivity_profile", required=True)
    parser.add_argument("--mass_config", default=None)
    parser.add_argument("--isotropic_sigma", type=float, default=0.0005)
    parser.add_argument("--modular_radii", type=_float_csv, default=_float_csv(DEFAULT_RADII))
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--candidate_pool_size", type=int, default=100)
    parser.add_argument("--candidate_prefix", type=int, default=25)
    parser.add_argument("--num_examples", type=int, default=64)
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Safety cap; prompts longer than this fail instead of being truncated",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--precision", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--power_iterations", type=int, default=8)
    parser.add_argument("--cuda_devices", default="0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--capture_layerwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.isotropic_sigma <= 0.0:
        parser.error("--isotropic_sigma must be positive")
    if args.candidate_pool_size < 1:
        parser.error("--candidate_pool_size must be positive")
    if not 1 <= args.candidate_prefix <= args.candidate_pool_size:
        parser.error("candidate prefix must lie in [1, candidate_pool_size]")
    if args.num_examples < 1 or args.max_length < 2 or args.batch_size < 1:
        parser.error(
            "num_examples/batch_size must be positive and max_length must be >=2"
        )
    if args.tp != 1:
        parser.error("The audited HF/vLLM physical-tensor bridge currently requires --tp 1")
    args.mass_config = load_mass_config(args.mass_config)
    args.sensitivity_profile = load_sensitivity_profile(args.sensitivity_profile)
    return args


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _unwrap(payload: Any) -> Any:
    current = payload
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]
    return current


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _git_metadata() -> Mapping[str, Any]:
    def command(*parts):
        try:
            return subprocess.check_output(
                parts, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = command("git", "status", "--porcelain=v1")
    diff = command("git", "diff", "--binary", "HEAD")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": _sha256_bytes((status or "").encode("utf-8")),
        "tracked_diff_sha256": _sha256_bytes((diff or "").encode("utf-8")),
    }


def _validate_profile(args, config, adapter) -> None:
    profile = args.sensitivity_profile
    architecture = profile.get("architecture", {})
    if architecture.get("family") != adapter.family:
        raise ValueError(
            f"Profile family {architecture.get('family')!r} does not match "
            f"runtime family {adapter.family!r}"
        )
    if architecture.get("model_type") != config.model_type:
        raise ValueError("Profile model_type does not match runtime model_type")
    if profile.get("model_name") != args.model_name:
        raise ValueError(
            "Profile model_name must exactly match --model_name"
        )
    expected_layers = int(getattr(effective_text_config(config), "num_hidden_layers", 0))
    if int(profile.get("num_layers", -1)) != expected_layers:
        raise ValueError("Profile layer count does not match runtime model config")


def _environment() -> Mapping[str, Any]:
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "ray": version("ray"),
        "vllm": version("vllm"),
        "transformers": version("transformers"),
    }


def _validate_runtime_layout(config, adapter, diagnostics) -> Mapping[str, Any]:
    """Assert the audited TP=1 physical-role pattern used by vLLM."""

    rows = list(diagnostics.get("parameters", []))
    fallback_roles = {"other", "attention_other", "mlp_other", "layer_norm_other"}
    fallback_parameters = [
        row.get("parameter_name")
        for row in rows
        if row.get("role") in fallback_roles
    ]
    if fallback_parameters:
        raise RuntimeError(
            "Runtime contains fallback parameter roles: "
            + ", ".join(str(name) for name in fallback_parameters[:10])
        )
    if len({row.get("parameter_name") for row in rows}) != len(rows):
        raise RuntimeError("Runtime layout contains duplicate physical parameter names")

    role_counts: dict[str, int] = {}
    for row in rows:
        role = str(row.get("role"))
        role_counts[role] = role_counts.get(role, 0) + 1

    if adapter.family == "llama":
        num_layers = int(effective_text_config(config).num_hidden_layers)
        expected = {
            "qkv": num_layers,
            "o": num_layers,
            "gate_up": num_layers,
            "down": num_layers,
            "input_norm": num_layers,
            "post_attention_norm": num_layers,
            "embedding": 1,
            "final_norm": 1,
        }
        if not bool(getattr(effective_text_config(config), "tie_word_embeddings", False)):
            expected["lm_head"] = 1
        if role_counts != expected:
            raise RuntimeError(
                f"Unexpected Llama TP=1 role counts: actual={role_counts}, "
                f"expected={expected}"
            )

        expected_layer_roles = {
            "qkv",
            "o",
            "gate_up",
            "down",
            "input_norm",
            "post_attention_norm",
        }
        for layer_index in range(num_layers):
            actual = {
                str(row.get("role"))
                for row in rows
                if row.get("layer_index") == layer_index
            }
            if actual != expected_layer_roles:
                raise RuntimeError(
                    f"Layer {layer_index} role mismatch: actual={actual}, "
                    f"expected={expected_layer_roles}"
                )
        for row in rows:
            role = row.get("role")
            multiplicity = float(row.get("multiplicity", 1.0))
            expected_multiplicity = 3.0 if role == "qkv" else 2.0 if role == "gate_up" else 1.0
            if multiplicity != expected_multiplicity:
                raise RuntimeError(
                    f"Unexpected multiplicity for {row.get('parameter_name')}: "
                    f"{multiplicity} != {expected_multiplicity}"
                )
    elif adapter.family == "gemma3":
        text_config = effective_text_config(config)
        num_layers = int(text_config.num_hidden_layers)
        per_layer_roles = {
            "qkv", "o", "gate_up", "down", "input_norm", "q_norm", "k_norm",
            "attention_output_norm", "pre_feedforward_norm", "mlp_output_norm",
        }
        expected = {role: num_layers for role in per_layer_roles}
        expected.update({"embedding": 1, "final_norm": 1})
        if not bool(getattr(text_config, "tie_word_embeddings", False)):
            expected["lm_head"] = 1
        if role_counts != expected:
            raise RuntimeError(
                f"Unexpected Gemma3 TP=1 role counts: actual={role_counts}, "
                f"expected={expected}"
            )
        for layer_index in range(num_layers):
            actual = {
                str(row.get("role"))
                for row in rows
                if row.get("layer_index") == layer_index
            }
            if actual != per_layer_roles:
                raise RuntimeError(
                    f"Gemma3 layer {layer_index} role mismatch: actual={actual}, "
                    f"expected={per_layer_roles}"
                )
        for row in rows:
            role = row.get("role")
            multiplicity = float(row.get("multiplicity", 1.0))
            expected_multiplicity = (
                3.0 if role == "qkv" else 2.0 if role == "gate_up" else 1.0
            )
            if multiplicity != expected_multiplicity:
                raise RuntimeError(
                    f"Unexpected multiplicity for {row.get('parameter_name')}: "
                    f"{multiplicity} != {expected_multiplicity}"
                )
    elif adapter.family == "olmo3":
        text_config = effective_text_config(config)
        num_layers = int(text_config.num_hidden_layers)
        per_layer_roles = {
            "qkv",
            "o",
            "gate_up",
            "down",
            "q_norm",
            "k_norm",
            "attention_output_norm",
            "mlp_output_norm",
        }
        expected = {role: num_layers for role in per_layer_roles}
        expected.update({"embedding": 1, "final_norm": 1})
        if not bool(getattr(text_config, "tie_word_embeddings", False)):
            expected["lm_head"] = 1
        if role_counts != expected:
            raise RuntimeError(
                f"Unexpected OLMo3 TP=1 role counts: actual={role_counts}, "
                f"expected={expected}"
            )
        for layer_index in range(num_layers):
            actual = {
                str(row.get("role"))
                for row in rows
                if row.get("layer_index") == layer_index
            }
            if actual != per_layer_roles:
                raise RuntimeError(
                    f"OLMo3 layer {layer_index} role mismatch: actual={actual}, "
                    f"expected={per_layer_roles}"
                )
        for row in rows:
            role = row.get("role")
            multiplicity = float(row.get("multiplicity", 1.0))
            expected_multiplicity = (
                3.0 if role == "qkv" else 2.0 if role == "gate_up" else 1.0
            )
            if multiplicity != expected_multiplicity:
                raise RuntimeError(
                    f"Unexpected multiplicity for {row.get('parameter_name')}: "
                    f"{multiplicity} != {expected_multiplicity}"
                )
    else:
        raise RuntimeError(
            "The functional-displacement implementation does not yet validate "
            f"to the audited Llama layout, received family={adapter.family!r}"
        )

    if int(diagnostics.get("num_active_parameters", -1)) != len(rows):
        raise RuntimeError("Not every runtime physical tensor is active")
    if int(diagnostics.get("num_unclassified_parameters", -1)) != 0:
        raise RuntimeError("Runtime reports unclassified physical tensors")
    return {"role_counts": role_counts, "expected_role_counts": expected}


def _precision_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _token_batches(tokenizer, input_ids_list, batch_size: int, device: torch.device):
    batches = []
    for start in range(0, len(input_ids_list), batch_size):
        rows = input_ids_list[start : start + batch_size]
        padded = tokenizer.pad(
            {"input_ids": rows},
            padding=True,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded["attention_mask"].to(device)
        prediction_mask = attention_mask.to(dtype=torch.bool).clone()
        for row_index in range(prediction_mask.shape[0]):
            valid = torch.nonzero(prediction_mask[row_index], as_tuple=False).flatten()
            if valid.numel() < 2:
                raise ValueError("Every diagnostic prompt must contain at least two tokens")
            prediction_mask[row_index, valid[-1]] = False
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prediction_mask": prediction_mask,
            }
        )
    return batches


def _forward_transformers(model, batch, *, capture_layerwise: bool):
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            # Final pre-head hidden states are always part of the objective;
            # the flag only controls whether intermediate layers are retained.
            output_hidden_states=True,
            return_dict=True,
        )
    if outputs.logits.ndim != 3:
        raise RuntimeError(f"Expected [batch,tokens,vocab] logits, got {outputs.logits.shape}")
    return {
        "logits": outputs.logits.detach(),
        "hidden": outputs.hidden_states[-1].detach(),
        "layerwise_hidden": (
            tuple(hidden.detach() for hidden in outputs.hidden_states[1:])
            if capture_layerwise
            else ()
        ),
    }


def _prepare_transformers_reference(model, batches, *, capture_layerwise: bool):
    cache = []
    for batch in batches:
        cache.append(
            _forward_transformers(
                model,
                batch,
                capture_layerwise=capture_layerwise,
            )
        )
    return cache


def _measure_transformers_candidate(
    model,
    batches,
    base_cache,
    *,
    capture_layerwise: bool,
) -> Mapping[str, Any]:
    kl_sum = 0.0
    position_count = 0
    hidden_delta_sq = 0.0
    hidden_base_sq = 0.0
    hidden_count = 0
    layer_accumulators: dict[int, dict[str, float]] = {}

    for batch, base in zip(batches, base_cache):
        current = _forward_transformers(
            model,
            batch,
            capture_layerwise=capture_layerwise,
        )
        mask = batch["prediction_mask"]
        positions = int(mask.sum().item())
        mean_kl = symmetric_kl_from_logits(
            base["logits"], current["logits"], mask=mask
        )
        kl_sum += mean_kl * positions
        position_count += positions

        base_hidden = base["hidden"][mask].double()
        current_hidden = current["hidden"][mask].double()
        delta = current_hidden - base_hidden
        hidden_delta_sq += float(delta.square().sum().item())
        hidden_base_sq += float(base_hidden.square().sum().item())
        hidden_count += int(base_hidden.numel())

        for layer_index, (base_layer, current_layer) in enumerate(
            zip(base["layerwise_hidden"], current["layerwise_hidden"])
        ):
            base_selected = base_layer[mask].double()
            current_selected = current_layer[mask].double()
            accumulator = layer_accumulators.setdefault(
                layer_index, {"delta_sq": 0.0, "base_sq": 0.0}
            )
            accumulator["delta_sq"] += float(
                (current_selected - base_selected).square().sum().item()
            )
            accumulator["base_sq"] += float(base_selected.square().sum().item())
        del current

    if position_count <= 0 or hidden_count <= 0:
        raise RuntimeError("Functional diagnostic produced no valid positions")
    return {
        "symmetric_kl": kl_sum / position_count,
        "hidden_absolute_rms": (hidden_delta_sq / hidden_count) ** 0.5,
        "hidden_relative_rms": (
            hidden_delta_sq / max(hidden_base_sq, 1e-30)
        ) ** 0.5,
        "num_teacher_forced_positions": position_count,
        "layerwise_hidden_relative_rms": {
            str(layer_index): (
                values["delta_sq"] / max(values["base_sq"], 1e-30)
            ) ** 0.5
            for layer_index, values in sorted(layer_accumulators.items())
        },
    }


def run(args) -> Mapping[str, Any]:
    source_manifest_at_start = source_manifest(REPO_ROOT)
    environment_at_start = _environment()
    git_at_start = _git_metadata()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    config = AutoConfig.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    is_local_model = Path(args.model_name).exists()
    resolved_model_revision = (
        args.model_revision if is_local_model else getattr(config, "_commit_hash", None)
    )
    if resolved_model_revision is None and not Path(args.model_name).exists():
        raise ValueError(
            "Could not resolve an immutable model revision"
        )
    adapter = get_architecture_adapter("auto", config.model_type)
    _validate_profile(args, config, adapter)
    profile_revision = args.sensitivity_profile.get("model_revision")
    if profile_revision != resolved_model_revision:
        raise ValueError(
            "Sensitivity profile was not calibrated from the resolved model revision"
        )
    calibration_tokenization = args.sensitivity_profile.get("calibration", {}).get(
        "prompt_tokenization", {}
    )
    if calibration_tokenization.get("scheme") != PROMPT_TOKENIZATION_SCHEME:
        raise ValueError(
            "Sensitivity profile was not calibrated with the official RandOpt "
            "rendered-prompt tokenization policy; regenerate the profile"
        )

    handler = get_dataset_handler(args.dataset)
    data_path = args.train_data_path or handler.default_train_path
    train_data_sha256_at_start = _path_fingerprint(data_path)
    if args.sensitivity_profile.get("dataset") != args.dataset:
        raise ValueError("Sensitivity profile dataset differs from the diagnostic")
    if args.sensitivity_profile.get("data_sha256") != train_data_sha256_at_start:
        raise ValueError(
            "Sensitivity profile and functional diagnostic use different data"
        )
    rows = handler.load_data(data_path, split="train", max_samples=args.num_examples)
    if len(rows) != args.num_examples:
        raise ValueError(f"Requested {args.num_examples} examples, loaded {len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=resolved_model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompts = render_prompts(
        tokenizer,
        args.model_name,
        rows,
        chat_template_date=args.chat_template_date,
    )
    input_ids_list = encode_rendered_prompts(
        tokenizer, prompts, max_length=args.max_length
    )
    if any(len(input_ids) < 2 for input_ids in input_ids_list):
        raise ValueError("Every diagnostic prompt must contain at least two tokens")
    prompt_manifest = build_prompt_manifest(
        tokenizer,
        input_ids_list,
        chat_template_date=args.chat_template_date,
    )
    prompt_manifest.update({
        "max_length": args.max_length,
        "token_lengths": [len(input_ids) for input_ids in input_ids_list],
        "teacher_forced_positions": "all prompt positions except final token",
        "formatting": "dataset messages -> model chat template with generation prompt",
        "tokenization": "tokenizer.encode(add_special_tokens=False) after one chat-template render",
        "truncation": "none; max_length is a fail-fast safety cap",
    })

    pool = build_candidate_seeds(args.global_seed, args.candidate_pool_size)
    candidate_seeds = pool[: args.candidate_prefix]

    # First build and validate the exact TP=1 physical tensor layout used by
    # the subsequent vLLM population search.  Functional forward passes are
    # intentionally not performed through a raw vLLM model call: without
    # scheduler attention metadata that path is only a profiling/dummy pass.
    engines = []
    pgs = []
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)
    try:
        engines, pgs = launch_engines(
            1,
            args.model_name,
            precision=args.precision,
            tensor_parallel_size=1,
            revision=resolved_model_revision,
        )
        profile_fingerprint = sensitivity_profile_fingerprint(args.sensitivity_profile)
        raw_diagnostics = ray.get(
            engines[0].collective_rpc.remote(
                "get_recursive_modular_diagnostics",
                args=(
                    args.mass_config,
                    "recursive_modular_shell_v2",
                    args.sensitivity_profile,
                    args.power_iterations,
                ),
            )
        )
        diagnostics = _unwrap(raw_diagnostics)
        layout_validation = _validate_runtime_layout(config, adapter, diagnostics)
        runtime_rows = list(diagnostics["parameters"])
    finally:
        cleanup_engines(engines, pgs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_kwargs = {
        "torch_dtype": _precision_dtype(args.precision),
        "attn_implementation": "eager",
    }
    model_class = transformers_causal_model_class(config)
    try:
        model = model_class.from_pretrained(
            args.model_name,
            revision=resolved_model_revision,
            **load_kwargs,
        )
    except TypeError:
        load_kwargs.pop("attn_implementation")
        model = model_class.from_pretrained(
            args.model_name,
            revision=resolved_model_revision,
            **load_kwargs,
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    bindings, parameters = build_physical_parameter_bindings(model, runtime_rows)
    base_snapshot = snapshot_parameters(parameters)
    batches = _token_batches(tokenizer, input_ids_list, args.batch_size, device)
    base_cache = _prepare_transformers_reference(
        model,
        batches,
        capture_layerwise=args.capture_layerwise,
    )

    def measure_candidate(seed: int, radius: float, method: str) -> Mapping[str, Any]:
        apply_physical_perturbation(
            bindings=bindings,
            parameters=parameters,
            candidate_seed=seed,
            radius=radius,
            method=method,
            power_iterations=args.power_iterations,
            restore=False,
        )
        try:
            metrics = dict(
                _measure_transformers_candidate(
                    model,
                    batches,
                    base_cache,
                    capture_layerwise=args.capture_layerwise,
                )
            )
            metrics.update(
                {"seed": int(seed), "radius": float(radius), "method": method}
            )
            return metrics
        finally:
            # Official RandOpt restores by regenerating the same candidate
            # noise and subtracting it; there is no per-candidate snapshot
            # reset in this diagnostic.
            apply_physical_perturbation(
                bindings=bindings,
                parameters=parameters,
                candidate_seed=seed,
                radius=radius,
                method=method,
                power_iterations=args.power_iterations,
                restore=True,
            )

    def measure_condition(radius: float, method: str):
        # Population runs load a fresh base checkpoint for each method/radius.
        # Initialize the whole condition once, then preserve official
        # add/generate/subtract behavior across its ordered candidates.
        initialize_condition_from_base(parameters, base_snapshot)
        candidate_rows = []
        for index, seed in enumerate(candidate_seeds):
            print(
                f"{method} radius={radius:g} candidate "
                f"{index + 1}/{len(candidate_seeds)}"
            )
            candidate_rows.append(measure_candidate(seed, radius, method))
        return candidate_rows, dict(parameter_drift_summary(parameters, base_snapshot))

    isotropic_rows, isotropic_restore_drift = measure_condition(
        args.isotropic_sigma, "isotropic"
    )
    isotropic = dict(aggregate_candidate_metrics(isotropic_rows))

    radius_rows = []
    for radius in args.modular_radii:
        candidate_rows, restore_drift = measure_condition(
            radius, "recursive_modular_shell_v2"
        )
        aggregate = dict(aggregate_candidate_metrics(candidate_rows))
        objective = radius_objective(
            aggregate["symmetric_kl"],
            isotropic["symmetric_kl"],
            aggregate["hidden_relative_rms"],
            isotropic["hidden_relative_rms"],
        )
        radius_rows.append(
            {
                "radius": radius,
                "objective_j": objective,
                "aggregate": aggregate,
                "candidates": candidate_rows,
                "restore_drift_after_prefix": restore_drift,
            }
        )
        partial = {
            "status": "running",
            "completed_radii": len(radius_rows),
            "isotropic": {"aggregate": isotropic, "candidates": isotropic_rows},
            "modular_radii": radius_rows,
        }
        _json_dump(args.output.with_suffix(args.output.suffix + ".partial"), partial)

    selected = select_radius(radius_rows)
    payload = {
            "schema_version": 2,
            "status": "complete",
            "selection_uses_accuracy": False,
            "model": {
                "name": args.model_name,
                "model_type": config.model_type,
                "architecture_family": adapter.family,
                "num_hidden_layers": getattr(
                    effective_text_config(config), "num_hidden_layers", None
                ),
                "tie_word_embeddings": getattr(
                    effective_text_config(config), "tie_word_embeddings", None
                ),
                "requested_revision": args.model_revision,
                "resolved_commit_hash": resolved_model_revision,
                "config_sha256": _sha256_bytes(
                    json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
                ),
            },
            "dataset": {
                "name": args.dataset,
                "train_data_path": str(data_path),
                "train_data_sha256": train_data_sha256_at_start,
            },
            "profile_fingerprint": profile_fingerprint,
            "mass_config": args.mass_config,
            "execution_protocol": "official_randopt_j_v1",
            "candidate_seed_scheme": CANDIDATE_SEED_SCHEME,
            "noise_scheme": PARAMETER_NOISE_SCHEME,
            "weight_restore_scheme": WEIGHT_RESTORE_SCHEME,
            "per_candidate_exact_reset": False,
            "condition_initialization": (
                "checkpoint-snapshot-copy-once-before-each-method-or-radius"
            ),
            "precision": args.precision,
            "tensor_parallel_size": args.tp,
            "power_iterations": args.power_iterations,
            "global_seed": args.global_seed,
            "candidate_pool_size": args.candidate_pool_size,
            "candidate_pool": pool,
            "candidate_prefix": args.candidate_prefix,
            "candidate_seeds": candidate_seeds,
            "prompt_manifest": prompt_manifest,
            "measurement_backend": {
                "forward": "transformers-eager-teacher-forced",
                "perturbation_layout": "vllm-tp1-physical-tensors",
                "fused_tensor_bridge": "qkv-and-gate-up-axis0-v1",
                "batch_size": args.batch_size,
                "capture_layerwise": bool(args.capture_layerwise),
            },
            "metric_aggregation": "arithmetic mean over the fixed candidate prefix",
            "runtime_layout": {
                "num_parameters": diagnostics.get("num_parameters"),
                "num_active_parameters": diagnostics.get("num_active_parameters"),
                "num_unclassified_parameters": diagnostics.get("num_unclassified_parameters"),
                "group_statistics": diagnostics.get("group_statistics"),
                "v2_to_v1_ratio_statistics": diagnostics.get(
                    "v2_to_v1_ratio_statistics"
                ),
                "layer_ratio_diagnostics": diagnostics.get(
                    "layer_ratio_diagnostics"
                ),
                "role_counts": layout_validation["role_counts"],
                "expected_role_counts": layout_validation["expected_role_counts"],
                "physical_parameter_bindings": binding_manifest(bindings),
            },
            "isotropic": {
                "sigma": args.isotropic_sigma,
                "aggregate": isotropic,
                "candidates": isotropic_rows,
                "restore_drift_after_prefix": isotropic_restore_drift,
            },
            "modular_radii": radius_rows,
            "objective": (
                "log(KL_mod/KL_iso)^2 + "
                "log(relative_hidden_RMS_mod/relative_hidden_RMS_iso)^2"
            ),
            "selected_radius": float(selected["radius"]),
            "selected_objective_j": float(selected["objective_j"]),
            "tie_break": "smallest_radius",
            "git": git_at_start,
            "source_manifest": source_manifest_at_start,
            "environment": environment_at_start,
    }
    if source_manifest(REPO_ROOT) != source_manifest_at_start:
        raise RuntimeError(
            "Functional-displacement source files changed while displacement was measured"
        )
    if _path_fingerprint(data_path) != train_data_sha256_at_start:
        raise RuntimeError(
            "Functional-calibration data changed while displacement was measured"
        )
    _json_dump(args.output, payload)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    print(f"Selected radius: {payload['selected_radius']}")
    print(f"Wrote: {args.output}")
    return payload


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
