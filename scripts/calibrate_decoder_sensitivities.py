#!/usr/bin/env python3
"""Architecture-aware sensitivity calibration for decoder-only language models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_handlers import get_dataset_handler, list_datasets  # noqa: E402
from scripts.calibrate_qwen_sensitivities import (  # noqa: E402
    _append_raw,
    _dtype,
    _estimate_one,
    _layer_indices,
    _spectral_norm_power,
)
from utils.official_prompt_protocol import (  # noqa: E402
    DEFAULT_CHAT_TEMPLATE_DATE,
    PROMPT_TOKENIZATION_SCHEME,
    encode_rendered_prompts,
    prompt_manifest,
    render_prompts,
)
from utils.architecture_adapters import get_architecture_adapter  # noqa: E402
from utils.architecture_adapters.base import detach_tree, extract_hidden  # noqa: E402
from utils.sensitivity_calibration import (  # noqa: E402
    aggregate_log_quantile,
    summarize_values,
)


PROFILE_SOURCE_PATHS = (
    "scripts/calibrate_decoder_sensitivities.py",
    "scripts/calibrate_qwen_sensitivities.py",
    "utils/official_prompt_protocol.py",
    "utils/sensitivity_calibration.py",
    "utils/architecture_adapters/__init__.py",
    "utils/architecture_adapters/base.py",
    "utils/architecture_adapters/gemma3.py",
    "utils/architecture_adapters/llama.py",
    "utils/architecture_adapters/olmo3.py",
    "utils/architecture_adapters/qwen2.py",
)


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


def _source_manifest() -> Mapping[str, Any]:
    files = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in PROFILE_SOURCE_PATHS
    }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"combined_sha256": combined, "files": files}


def parse_args(
    argv=None,
    *,
    forced_architecture: str | None = None,
    legacy_qwen_schema: bool = False,
):
    parser = argparse.ArgumentParser(
        description="Calibrate an architecture-aware decoder sensitivity profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--architecture",
        choices=("auto", "qwen2", "llama", "olmo3", "gemma3"),
        default="auto",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--dataset", default="countdown", choices=list_datasets())
    parser.add_argument("--data_path", default=None)
    parser.add_argument(
        "--chat_template_date", default=DEFAULT_CHAT_TEMPLATE_DATE
    )
    parser.add_argument("--num_examples", type=int, default=64)
    parser.add_argument("--layers_per_example", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--estimator", choices=("power", "directional"), default="power")
    parser.add_argument("--jvp_power_iterations", type=int, default=3)
    parser.add_argument("--directional_probes", type=int, default=3)
    parser.add_argument("--estimator_seed", type=int, default=1729)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--aggregate_quantile", type=float, default=0.90)
    parser.add_argument("--clip_min", type=float, default=0.25)
    parser.add_argument("--clip_max", type=float, default=4.0)
    parser.add_argument("--factor_clip_min", type=float, default=0.25)
    parser.add_argument("--factor_clip_max", type=float, default=4.0)
    parser.add_argument("--linear_policy", choices=("analytic", "unit"), default="analytic")
    parser.add_argument("--scale_normalization", choices=("match_v1_median", "unit_median", "none"), default="none")
    parser.add_argument("--block_reconciliation", choices=("geometric", "none"), default="none")
    parser.add_argument("--local_ratio_clip_min", type=float, default=0.5)
    parser.add_argument("--local_ratio_clip_max", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument("--continue_on_estimator_error", action="store_true")
    parser.add_argument("--legacy_qwen_schema", action="store_true", default=legacy_qwen_schema)
    args = parser.parse_args(argv)
    if forced_architecture is not None:
        args.architecture = forced_architecture
    if args.num_examples < 1 or args.layers_per_example < 1:
        parser.error("num_examples and layers_per_example must be positive")
    if args.max_length < 2:
        parser.error("max_length must be at least 2")
    if args.jvp_power_iterations < 1 or args.directional_probes < 1:
        parser.error("estimator iteration/probe counts must be positive")
    if not 0.0 <= args.aggregate_quantile <= 1.0:
        parser.error("aggregate_quantile must be in [0, 1]")
    if not 0.0 < args.clip_min <= args.clip_max:
        parser.error("clip range must satisfy 0 < min <= max")
    if not 0.0 < args.factor_clip_min <= args.factor_clip_max:
        parser.error("factor clip range must satisfy 0 < min <= max")
    if not 0.25 <= args.local_ratio_clip_min <= 1.0:
        parser.error("local_ratio_clip_min must lie in [0.25, 1]")
    if not 1.0 <= args.local_ratio_clip_max <= 4.0:
        parser.error("local_ratio_clip_max must lie in [1, 4]")
    return args


@contextmanager
def _capture_inputs(model, decoder, layers, adapter):
    captures: Dict[str, Any] = {}
    handles = []

    def register(module, key):
        def hook(_module, args, kwargs):
            captures[key] = {"args": detach_tree(args), "kwargs": detach_tree(kwargs)}

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    for layer_index, layer in enumerate(layers):
        for name, module in adapter.capture_modules(layer).items():
            register(module, f"layer.{layer_index}.{name}")
    if hasattr(decoder, "norm"):
        register(decoder.norm, "global.final_norm")
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def _output_head_profile(model):
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight") or head.weight.ndim != 2:
        return {"value": 1.0, "source": "fallback_unit"}
    weight = head.weight.detach().float()
    d_out, d_in = weight.shape
    sigma = _spectral_norm_power(weight, iterations=8)
    return {
        "value": max(math.sqrt(float(d_in) / float(d_out)) * sigma, 1e-12),
        "source": "power_iteration_rms_to_rms",
    }


def run(args) -> Dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers before running calibration") from exc

    source_manifest_at_start = _source_manifest()
    config = AutoConfig.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    resolved_model_revision = (
        args.model_revision
        if Path(args.model_name).exists()
        else getattr(config, "_commit_hash", None)
    )
    if resolved_model_revision is None and not Path(args.model_name).exists():
        raise ValueError("Could not resolve an immutable model revision")
    adapter = get_architecture_adapter(args.architecture, config.model_type)
    from utils.model_config import transformers_causal_model_class
    model_class = transformers_causal_model_class(config)
    if args.legacy_qwen_schema and adapter.family != "qwen2":
        raise ValueError("--legacy_qwen_schema is only valid for qwen2")

    random.seed(args.data_seed)
    torch.manual_seed(args.estimator_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.estimator_seed)

    handler = get_dataset_handler(args.dataset)
    data_path = args.data_path or handler.default_train_path
    data_sha256_at_start = _path_fingerprint(data_path)
    datas = handler.load_data(data_path, split="train", max_samples=args.num_examples)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=resolved_model_revision,
    )
    prompts = render_prompts(
        tokenizer,
        args.model_name,
        datas,
        chat_template_date=args.chat_template_date,
    )
    if len(prompts) < args.num_examples:
        print(f"WARNING: requested {args.num_examples}, loaded {len(prompts)} examples")

    print(
        f"Loading {args.model_name} as {adapter.family} on {args.device} "
        "with eager attention..."
    )
    load_kwargs = {"torch_dtype": _dtype(args.precision), "attn_implementation": "eager"}
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
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    decoder = adapter.get_decoder(model)
    layers = list(decoder.layers)
    if not layers:
        raise RuntimeError("Decoder has no layers")
    for layer_index, layer in enumerate(layers):
        adapter.validate_layer(layer, layer_index)

    output = args.output.expanduser().resolve()
    raw_output = (
        args.raw_output.expanduser().resolve()
        if args.raw_output is not None
        else output.with_name(output.stem + "_raw_samples.jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raw_output.unlink()

    samples: MutableMapping[Tuple[str, str], List[float]] = defaultdict(list)
    calibration_start = time.perf_counter()
    full_calibration_token_ids = encode_rendered_prompts(tokenizer, prompts)
    # Preserve the Qwen calibration protocol: use the leading fixed-size
    # window for sensitivity estimation, while J(r) and generation use the
    # complete prompt.
    calibration_token_ids = [
        input_ids[: args.max_length] for input_ids in full_calibration_token_ids
    ]
    encoded_prompts = [
        {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
        }
        for input_ids in calibration_token_ids
    ]
    calibration_prompt_manifest = prompt_manifest(
        tokenizer,
        calibration_token_ids,
        chat_template_date=args.chat_template_date,
    )
    for example_index, encoded in enumerate(encoded_prompts):
        inputs = {key: value.to(args.device) for key, value in encoded.items()}
        rows: List[Dict[str, Any]] = []
        with _capture_inputs(model, decoder, layers, adapter) as captures:
            with torch.no_grad():
                model(**inputs, use_cache=False)

        selected = _layer_indices(example_index, len(layers), args.layers_per_example)
        print(f"Example {example_index + 1}/{len(prompts)} | layers={selected}")
        for layer_index in selected:
            prefix = f"layer.{layer_index}."
            local = {
                key.removeprefix(prefix): value
                for key, value in captures.items()
                if key.startswith(prefix)
            }
            specs = adapter.metric_specs(layers[layer_index], local, layer_index)
            base_seed = args.estimator_seed + example_index * 100003 + layer_index * 101
            for metric_index, spec in enumerate(specs):
                _estimate_one(
                    samples,
                    rows,
                    args,
                    scope=str(layer_index),
                    metric=spec.name,
                    function=spec.function,
                    point=spec.point,
                    example_index=example_index,
                    layer_index=layer_index,
                    seed=base_seed + metric_index,
                )

        if "global.final_norm" in captures:
            _estimate_one(
                samples,
                rows,
                args,
                scope="global",
                metric="final_norm",
                function=decoder.norm,
                point=extract_hidden(captures["global.final_norm"]),
                example_index=example_index,
                layer_index=None,
                seed=args.estimator_seed + example_index * 100003 + 90001,
            )
        _append_raw(raw_output, rows)
        del inputs, encoded, captures, rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_profile: Dict[str, Any] = {}
    missing = []
    for layer_index, layer in enumerate(layers):
        metrics = {}
        for metric in adapter.required_metrics:
            values = samples.get((str(layer_index), metric), [])
            if not values:
                missing.append(f"layers.{layer_index}.metrics.{metric}")
                continue
            metrics[metric] = {
                "value": aggregate_log_quantile(values, args.aggregate_quantile),
                "summary": summarize_values(values),
            }
        if args.legacy_qwen_schema:
            layer_profile[str(layer_index)] = metrics
        else:
            layer_profile[str(layer_index)] = {
                "metadata": dict(adapter.layer_metadata(layer)),
                "metrics": metrics,
            }
    if missing:
        raise RuntimeError("Calibration profile is incomplete: " + ", ".join(missing[:20]))

    final_values = samples.get(("global", "final_norm"), [])
    global_profile = {
        "final_norm": (
            {
                "value": aggregate_log_quantile(final_values, args.aggregate_quantile),
                "summary": summarize_values(final_values),
            }
            if final_values
            else {"value": 1.0, "summary": {"count": 0}}
        ),
        "output_head": _output_head_profile(model),
    }
    metric_diagnostics: Dict[str, Any] = {}
    for (scope, metric), values in sorted(samples.items()):
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        key = f"{scope}.{metric}"
        metric_diagnostics[key] = {
            "scope": scope,
            "metric": metric,
            "summary": summarize_values(finite_values),
            "clip_min": args.clip_min,
            "clip_max": args.clip_max,
            "num_below_clip_min": sum(value < args.clip_min for value in finite_values),
            "num_above_clip_max": sum(value > args.clip_max for value in finite_values),
            "clip_fraction": (
                sum(
                    value < args.clip_min or value > args.clip_max
                    for value in finite_values
                )
                / len(finite_values)
                if finite_values
                else 0.0
            ),
        }
    profile = {
        "schema_version": 2 if args.legacy_qwen_schema else 3,
        "method": "recursive_modular_shell_v2",
        "model_name": args.model_name,
        "model_revision": resolved_model_revision,
        "dataset": args.dataset,
        "data_path": str(data_path),
        "data_sha256": data_sha256_at_start,
        "num_layers": len(layers),
        "calibration": {
            "num_examples": len(prompts),
            "layers_per_example": args.layers_per_example,
            "max_length": args.max_length,
            "precision": args.precision,
            "device": args.device,
            "estimator": args.estimator,
            "jvp_power_iterations": args.jvp_power_iterations,
            "directional_probes": args.directional_probes,
            "estimator_seed": args.estimator_seed,
            "data_seed": args.data_seed,
            "aggregate": "exp(quantile(log sensitivity))",
            "aggregate_quantile": args.aggregate_quantile,
            "elapsed_sec": time.perf_counter() - calibration_start,
            "raw_samples_path": str(raw_output),
            "metric_diagnostics": metric_diagnostics,
            "prompt_tokenization": {
                **calibration_prompt_manifest,
                "scheme": PROMPT_TOKENIZATION_SCHEME,
                "truncation": f"right truncation to leading max_length={args.max_length}",
                "token_lengths": [len(row) for row in calibration_token_ids],
            },
        },
        "application": {
            "clip_min": args.clip_min,
            "clip_max": args.clip_max,
            "factor_clip_min": args.factor_clip_min,
            "factor_clip_max": args.factor_clip_max,
            "linear_policy": args.linear_policy,
            "scale_normalization": args.scale_normalization,
            "block_reconciliation": args.block_reconciliation,
            "strict_profile": True,
            "propagation_mode": "local_only",
            "local_normalization": "per_block_median",
            "local_ratio_clip_min": args.local_ratio_clip_min,
            "local_ratio_clip_max": args.local_ratio_clip_max,
        },
        "global": global_profile,
        "layers": layer_profile,
        "source_manifest": source_manifest_at_start,
    }
    if not args.legacy_qwen_schema:
        profile["architecture"] = {
            "family": adapter.family,
            "model_type": config.model_type,
            "block_layout": adapter.block_layout,
        }
    if _path_fingerprint(data_path) != data_sha256_at_start:
        raise RuntimeError("Calibration data changed while the profile was generated")
    if _source_manifest() != source_manifest_at_start:
        raise RuntimeError("Calibration source files changed while the profile was generated")
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote profile: {output}")
    print(f"Wrote raw samples: {raw_output}")
    return profile


def main(argv=None, *, forced_architecture=None, legacy_qwen_schema=False):
    return run(
        parse_args(
            argv,
            forced_architecture=forced_architecture,
            legacy_qwen_schema=legacy_qwen_schema,
        )
    )


if __name__ == "__main__":
    main()
