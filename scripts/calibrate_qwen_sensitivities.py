#!/usr/bin/env python3
"""Calibrate a fixed Qwen sensitivity profile for Recursive Modular Shell V2.

The script freezes a Hugging Face causal LM, captures local activations on the
training split, and estimates local Jacobian operator norms for Qwen decoder
submodules.  The resulting JSON profile is consumed unchanged by vLLM RandOpt
workers; no candidate or validation data is used during calibration.

This implementation is designed for Qwen2/Qwen2.5-style decoder layers with:
    input_layernorm, self_attn, post_attention_layernorm, and mlp.
It intentionally loads eager attention because torch.func JVP/VJP through fused
attention kernels is not consistently supported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_handlers import get_dataset_handler, list_datasets  # noqa: E402
from utils.sensitivity_calibration import (  # noqa: E402
    aggregate_log_quantile,
    estimate_operator_norm,
    first_tensor,
    summarize_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a fixed Qwen local-sensitivity profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", default="countdown", choices=list_datasets())
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--num_examples", type=int, default=64)
    parser.add_argument("--layers_per_example", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
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
    parser.add_argument(
        "--linear_policy", choices=("analytic", "unit"), default="analytic"
    )
    parser.add_argument(
        "--scale_normalization",
        choices=("match_v1_median", "unit_median", "none"),
        default="match_v1_median",
    )
    parser.add_argument(
        "--block_reconciliation", choices=("geometric", "none"), default="geometric"
    )
    parser.add_argument(
        "--local_ratio_clip_min",
        type=float,
        default=0.5,
        help="Minimum final per-parameter V2/V1 scale ratio after per-block centering",
    )
    parser.add_argument(
        "--local_ratio_clip_max",
        type=float,
        default=2.0,
        help="Maximum final per-parameter V2/V1 scale ratio after per-block centering",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw_output", type=Path, default=None)
    parser.add_argument(
        "--continue_on_estimator_error",
        action="store_true",
        help="Record failed local estimates and continue; missing layer metrics still fail at aggregation",
    )
    args = parser.parse_args()
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




def _spectral_norm_power(weight: torch.Tensor, iterations: int = 8) -> float:
    matrix = weight.detach()
    if matrix.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        matrix = matrix.float()
    rows, cols = matrix.shape
    vector = torch.ones(cols, device=matrix.device, dtype=matrix.dtype)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    for _ in range(max(1, int(iterations))):
        left = torch.mv(matrix, vector)
        left = left / torch.linalg.vector_norm(left).clamp_min(1e-12)
        vector = torch.mv(matrix.transpose(0, 1), left)
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    return max(float(torch.linalg.vector_norm(torch.mv(matrix, vector)).item()), 1e-12)

def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _extract_hidden(args: Sequence[Any], kwargs: Mapping[str, Any]) -> torch.Tensor:
    value = kwargs.get("hidden_states")
    if isinstance(value, torch.Tensor):
        return value
    for item in args:
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            return item
    raise RuntimeError("Could not identify hidden_states in module inputs")


def _sanitize_attention_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    clean = dict(_detach_tree(kwargs))
    clean.pop("hidden_states", None)
    clean.pop("past_key_value", None)
    clean.pop("past_key_values", None)
    clean["use_cache"] = False
    clean["output_attentions"] = False
    return clean


def _call_attention(module: torch.nn.Module, hidden: torch.Tensor, kwargs: Mapping[str, Any]):
    call_kwargs = dict(kwargs)
    try:
        return first_tensor(module(hidden_states=hidden, **call_kwargs))
    except TypeError:
        call_kwargs.pop("use_cache", None)
        call_kwargs.pop("output_attentions", None)
        try:
            return first_tensor(module(hidden_states=hidden, **call_kwargs))
        except TypeError:
            return first_tensor(module(hidden, **call_kwargs))


def _base_decoder(model: torch.nn.Module) -> torch.nn.Module:
    candidates = [
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "base_model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    if hasattr(model, "layers"):
        return model
    raise RuntimeError("Could not locate decoder layers on the loaded model")


def _layer_indices(example_index: int, num_layers: int, layers_per_example: int) -> List[int]:
    count = min(num_layers, layers_per_example)
    start = (example_index * count) % num_layers
    return [(start + offset) % num_layers for offset in range(count)]


def _format_prompts(tokenizer, model_name: str, datas: Sequence[Mapping[str, Any]]) -> List[str]:
    instruct = any(token in model_name.lower() for token in ("instruct", "chat", "-it"))
    prompts = []
    for data in datas:
        messages = data["messages"]
        if instruct and tokenizer.chat_template:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        else:
            prompts.append("\n".join(message["content"] for message in messages) + "\n")
    return prompts


@contextmanager
def _capture_qwen_inputs(model: torch.nn.Module, layers: Sequence[torch.nn.Module]):
    captures: Dict[str, Any] = {}
    handles = []

    def register(module: torch.nn.Module, key: str):
        def hook(_module, args, kwargs):
            captures[key] = {
                "args": _detach_tree(args),
                "kwargs": _detach_tree(kwargs),
            }

        try:
            handle = module.register_forward_pre_hook(hook, with_kwargs=True)
        except TypeError as exc:
            raise RuntimeError(
                "Calibration requires a PyTorch version supporting with_kwargs=True hooks"
            ) from exc
        handles.append(handle)

    for index, layer in enumerate(layers):
        register(layer, f"layer.{index}")
        register(layer.self_attn, f"layer.{index}.self_attn")
        register(layer.post_attention_layernorm, f"layer.{index}.post_norm")
        register(layer.mlp, f"layer.{index}.mlp")

    decoder = _base_decoder(model)
    if hasattr(decoder, "norm"):
        register(decoder.norm, "global.final_norm")
    head = getattr(model, "lm_head", None)
    if isinstance(head, torch.nn.Module):
        register(head, "global.lm_head")

    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def _estimate(
    function,
    point: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> float:
    return estimate_operator_norm(
        function,
        point,
        estimator=args.estimator,
        power_iterations=args.jvp_power_iterations,
        directions=args.directional_probes,
        seed=seed,
    )


def _append_raw(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def _record_estimate(
    samples: MutableMapping[Tuple[str, str], List[float]],
    rows: List[Dict[str, Any]],
    *,
    scope: str,
    metric: str,
    value: float,
    example_index: int,
    layer_index: int | None,
    elapsed_sec: float,
) -> None:
    key = (scope, metric)
    samples[key].append(float(value))
    rows.append(
        {
            "example_index": example_index,
            "layer_index": layer_index,
            "scope": scope,
            "metric": metric,
            "value": float(value),
            "elapsed_sec": float(elapsed_sec),
        }
    )


def _estimate_one(
    samples,
    rows,
    args,
    *,
    scope,
    metric,
    function,
    point,
    example_index,
    layer_index,
    seed,
):
    started = time.perf_counter()
    try:
        value = _estimate(function, point, args, seed)
    except Exception as exc:
        rows.append(
            {
                "example_index": example_index,
                "layer_index": layer_index,
                "scope": scope,
                "metric": metric,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": time.perf_counter() - started,
            }
        )
        if not args.continue_on_estimator_error:
            raise
        print(f"WARNING: failed {scope}.{metric}: {type(exc).__name__}: {exc}")
        return
    _record_estimate(
        samples,
        rows,
        scope=scope,
        metric=metric,
        value=value,
        example_index=example_index,
        layer_index=layer_index,
        elapsed_sec=time.perf_counter() - started,
    )


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers before running calibration") from exc

    random.seed(args.data_seed)
    torch.manual_seed(args.estimator_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.estimator_seed)

    handler = get_dataset_handler(args.dataset)
    data_path = args.data_path or handler.default_train_path
    datas = handler.load_data(data_path, split="train", max_samples=args.num_examples)
    if len(datas) < args.num_examples:
        print(f"WARNING: requested {args.num_examples}, loaded {len(datas)} examples")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompts = _format_prompts(tokenizer, args.model_name, datas)

    print(f"Loading {args.model_name} on {args.device} with eager attention...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=_dtype(args.precision),
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=_dtype(args.precision),
        )
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    decoder = _base_decoder(model)
    layers = list(decoder.layers)
    if not layers:
        raise RuntimeError("Decoder has no layers")
    for index, layer in enumerate(layers):
        for attribute in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        ):
            if not hasattr(layer, attribute):
                raise RuntimeError(f"Layer {index} lacks required attribute {attribute}")

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

    for example_index, prompt in enumerate(prompts):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {key: value.to(args.device) for key, value in encoded.items()}
        rows: List[Dict[str, Any]] = []

        with _capture_qwen_inputs(model, layers) as captures:
            with torch.no_grad():
                model(**inputs, use_cache=False)

        selected_layers = _layer_indices(
            example_index,
            len(layers),
            args.layers_per_example,
        )
        print(
            f"Example {example_index + 1}/{len(prompts)} | "
            f"layers={selected_layers}"
        )

        for layer_index in selected_layers:
            layer = layers[layer_index]
            layer_capture = captures[f"layer.{layer_index}"]
            attn_capture = captures[f"layer.{layer_index}.self_attn"]
            post_capture = captures[f"layer.{layer_index}.post_norm"]
            mlp_capture = captures[f"layer.{layer_index}.mlp"]

            block_input = _extract_hidden(
                layer_capture["args"], layer_capture["kwargs"]
            )
            attn_input = _extract_hidden(
                attn_capture["args"], attn_capture["kwargs"]
            )
            attention_residual_stream = _extract_hidden(
                post_capture["args"], post_capture["kwargs"]
            )
            mlp_input = _extract_hidden(mlp_capture["args"], mlp_capture["kwargs"])
            attention_kwargs = _sanitize_attention_kwargs(attn_capture["kwargs"])
            scope = str(layer_index)
            base_seed = args.estimator_seed + example_index * 100003 + layer_index * 101

            attention_function = lambda hidden, module=layer.self_attn, kwargs=attention_kwargs: _call_attention(
                module, hidden, kwargs
            )
            attention_residual_function = (
                lambda hidden, layer=layer, kwargs=attention_kwargs: hidden
                + _call_attention(layer.self_attn, layer.input_layernorm(hidden), kwargs)
            )
            mlp_function = lambda hidden, module=layer.mlp: module(hidden)
            mlp_residual_function = (
                lambda hidden, layer=layer: hidden
                + layer.mlp(layer.post_attention_layernorm(hidden))
            )
            block_function = (
                lambda hidden, attn=attention_residual_function, mlp=mlp_residual_function: mlp(
                    attn(hidden)
                )
            )

            estimates = (
                ("input_norm", layer.input_layernorm, block_input),
                ("post_attention_norm", layer.post_attention_layernorm, attention_residual_stream),
                ("attention_module", attention_function, attn_input),
                ("attention_residual", attention_residual_function, block_input),
                ("mlp_module", mlp_function, mlp_input),
                ("mlp_residual", mlp_residual_function, attention_residual_stream),
                ("block", block_function, block_input),
            )
            for metric_index, (metric, function, point) in enumerate(estimates):
                _estimate_one(
                    samples,
                    rows,
                    args,
                    scope=scope,
                    metric=metric,
                    function=function,
                    point=point,
                    example_index=example_index,
                    layer_index=layer_index,
                    seed=base_seed + metric_index,
                )

        if "global.final_norm" in captures:
            final_capture = captures["global.final_norm"]
            final_input = _extract_hidden(final_capture["args"], final_capture["kwargs"])
            _estimate_one(
                samples,
                rows,
                args,
                scope="global",
                metric="final_norm",
                function=decoder.norm,
                point=final_input,
                example_index=example_index,
                layer_index=None,
                seed=args.estimator_seed + example_index * 100003 + 90001,
            )

        _append_raw(raw_output, rows)
        del inputs, encoded, captures, rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_profile: Dict[str, Any] = {}
    missing: List[str] = []
    for layer_index in range(len(layers)):
        row: Dict[str, Any] = {}
        for metric in (
            "input_norm",
            "post_attention_norm",
            "attention_module",
            "attention_residual",
            "mlp_module",
            "mlp_residual",
            "block",
        ):
            values = samples.get((str(layer_index), metric), [])
            if not values:
                missing.append(f"layers.{layer_index}.{metric}")
                continue
            value = aggregate_log_quantile(values, args.aggregate_quantile)
            row[metric] = {
                "value": value,
                "summary": summarize_values(values),
            }
        layer_profile[str(layer_index)] = row

    global_profile: Dict[str, Any] = {}
    final_values = samples.get(("global", "final_norm"), [])
    if final_values:
        global_profile["final_norm"] = {
            "value": aggregate_log_quantile(final_values, args.aggregate_quantile),
            "summary": summarize_values(final_values),
        }
    else:
        global_profile["final_norm"] = {"value": 1.0, "summary": {"count": 0}}

    head = getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight") and head.weight.ndim == 2:
        weight = head.weight.detach().float()
        d_out, d_in = weight.shape
        sigma = _spectral_norm_power(weight, iterations=8)
        output_head = math.sqrt(float(d_in) / float(d_out)) * float(sigma)
        global_profile["output_head"] = {
            "value": max(output_head, 1e-12),
            "source": "power_iteration_rms_to_rms",
        }
        del weight
    else:
        global_profile["output_head"] = {
            "value": 1.0,
            "source": "fallback_unit",
        }

    if missing:
        raise RuntimeError(
            "Calibration did not produce a complete profile. Missing: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    profile = {
        "schema_version": 2,
        "method": "recursive_modular_shell_v2",
        "model_name": args.model_name,
        "dataset": args.dataset,
        "data_path": str(data_path),
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
        "notes": [
            "The model was frozen and evaluated with eager attention.",
            "Only training-split examples were used.",
            "The profile is fixed before RandOpt candidate sampling.",
            "Qwen forward behavior is not modified by the V2 perturbation method.",
            "Residual and full-block sensitivities are diagnostics only in V2.1 local-only propagation.",
            "Final V2/V1 scale ratios are centered per block and bounded by the local ratio clip.",
        ],
    }
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote sensitivity profile: {output}")
    print(f"Wrote raw samples: {raw_output}")
    print(f"Elapsed: {profile['calibration']['elapsed_sec'] / 60.0:.2f} min")


if __name__ == "__main__":
    main()
