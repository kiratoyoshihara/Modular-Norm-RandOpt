"""Pure helpers for functional-distance-matched controls.

The control uses the forward KL from the base model to a perturbed model on a
fixed, base-generated token trajectory.  Scale matching is performed without
downstream accuracy: noisy calibration curves are first made monotone with
isotonic regression in log--log space, then inverted by interpolation.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

import torch


DISTANCE_MATCH_PROTOCOL = "functional_distance_matched_control_v1"


def directional_kl_per_position(
    base_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    *,
    position_chunk_size: int = 8,
) -> torch.Tensor:
    """Return ``KL(base || perturbed)`` for every token position.

    Inputs must have shape ``[positions, vocabulary]``.  Computation is done
    in float32 chunks to bound the temporary softmax memory.  The returned
    tensor is float64 on CPU so callers can retain exact summaries without
    keeping vocabulary-sized tensors alive.
    """

    if base_logits.shape != perturbed_logits.shape:
        raise ValueError("base and perturbed logits must have identical shapes")
    if base_logits.ndim != 2:
        raise ValueError("expected [positions, vocabulary] logits")
    if base_logits.shape[0] < 1 or base_logits.shape[1] < 2:
        raise ValueError("at least one position and two vocabulary entries are required")
    if position_chunk_size < 1:
        raise ValueError("position_chunk_size must be positive")

    values: list[torch.Tensor] = []
    for start in range(0, base_logits.shape[0], position_chunk_size):
        stop = min(start + position_chunk_size, base_logits.shape[0])
        base = base_logits[start:stop].float()
        perturbed = perturbed_logits[start:stop].float()
        probabilities = torch.softmax(base, dim=-1)
        logit_delta = perturbed - base
        mean_delta = (probabilities * logit_delta).sum(dim=-1, keepdim=True)
        centered_delta = logit_delta - mean_delta

        # If q = softmax(base + d), then
        #   KL(p || q) = log E_p[exp(d)] - E_p[d].
        # Centering d and evaluating exp(c) as 1 + c + remainder
        # avoids subtracting two nearly equal log-softmax tensors.  The
        # remainder exp(c) - 1 - c is analytically non-negative.
        centered_mean = (probabilities * centered_delta).sum(dim=-1)
        exponential_remainder = (
            torch.expm1(centered_delta) - centered_delta
        ).clamp_min(0.0)
        mean_remainder = (probabilities * exponential_remainder).sum(dim=-1)
        per_position = torch.log1p(centered_mean + mean_remainder) - centered_mean

        # Forward KL is non-negative.  Keep a float64 fallback for exceptional
        # rows (for example, overflow in expm1 under an extreme perturbation)
        # and report genuinely non-finite model output explicitly.
        suspect = ~torch.isfinite(per_position) | (per_position < 0.0)
        result = per_position.detach().double().cpu()
        if bool(suspect.any().item()):
            suspect_base = base[suspect]
            suspect_perturbed = perturbed[suspect]
            if not bool(
                torch.isfinite(suspect_base).all().item()
                and torch.isfinite(suspect_perturbed).all().item()
            ):
                raise ValueError(
                    "non-finite model logits encountered while computing output KL"
                )
            precise_log_p = torch.log_softmax(suspect_base.double(), dim=-1)
            precise_log_q = torch.log_softmax(suspect_perturbed.double(), dim=-1)
            precise = (
                precise_log_p.exp() * (precise_log_p - precise_log_q)
            ).sum(dim=-1)
            if not bool(torch.isfinite(precise).all().item()):
                raise ValueError("non-finite output KL after float64 recomputation")
            # A final sub-ulp negative is still possible in a finite sum.  It
            # represents zero because exact KL cannot be negative.
            precise = precise.clamp_min(0.0)
            result[suspect.detach().cpu()] = precise.detach().cpu()
        values.append(result)
    return torch.cat(values, dim=0)


def prediction_flip_per_position(
    base_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
) -> torch.Tensor:
    """Return a CPU boolean indicating whether each next-token argmax flips."""

    if base_logits.shape != perturbed_logits.shape:
        raise ValueError("base and perturbed logits must have identical shapes")
    if base_logits.ndim != 2:
        raise ValueError("expected [positions, vocabulary] logits")
    return (
        base_logits.argmax(dim=-1) != perturbed_logits.argmax(dim=-1)
    ).detach().cpu()


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated quantile with no NumPy dependency."""

    materialized = sorted(float(value) for value in values)
    if not materialized:
        raise ValueError("quantile input must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    if len(materialized) == 1:
        return materialized[0]
    position = probability * (len(materialized) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return materialized[lower]
    fraction = position - lower
    return materialized[lower] + fraction * (
        materialized[upper] - materialized[lower]
    )


def summarize_candidate_examples(
    example_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Apply the predeclared token -> example -> candidate aggregation.

    Each example row supplies token-level KL values and token-level flip
    indicators.  The primary candidate metric is the median across examples
    of the within-example median token KL.  This implements the hierarchical
    median requested by the control protocol rather than allowing examples
    with long generations to dominate.
    """

    if not example_rows:
        raise ValueError("candidate must contain at least one example")
    example_medians: list[float] = []
    example_means: list[float] = []
    example_flip_rates: list[float] = []
    all_kl: list[float] = []
    all_flips: list[float] = []
    for row in example_rows:
        kl_values = [float(value) for value in row["token_kl"]]
        flip_values = [float(bool(value)) for value in row["token_flips"]]
        if not kl_values or len(kl_values) != len(flip_values):
            raise ValueError("token KL and flip arrays must be non-empty and aligned")
        if any(not math.isfinite(value) or value < -1e-8 for value in kl_values):
            raise ValueError("token KL values must be finite and non-negative")
        kl_values = [max(0.0, value) for value in kl_values]
        example_medians.append(float(statistics.median(kl_values)))
        example_means.append(float(statistics.fmean(kl_values)))
        example_flip_rates.append(float(statistics.fmean(flip_values)))
        all_kl.extend(kl_values)
        all_flips.extend(flip_values)
    return {
        "median_output_kl": float(statistics.median(example_medians)),
        "mean_of_example_mean_output_kl": float(statistics.fmean(example_means)),
        "global_token_median_output_kl": float(statistics.median(all_kl)),
        "median_prediction_flip_rate": float(statistics.median(example_flip_rates)),
        "global_prediction_flip_rate": float(statistics.fmean(all_flips)),
        "num_examples": len(example_rows),
        "num_teacher_forced_tokens": len(all_kl),
    }


def summarize_condition_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Summarize a scale condition across its paired candidate seeds."""

    if not candidate_rows:
        raise ValueError("condition must contain at least one candidate")
    kl = [float(row["median_output_kl"]) for row in candidate_rows]
    flips = [float(row["median_prediction_flip_rate"]) for row in candidate_rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in kl):
        raise ValueError("candidate median KL values must be finite and positive")
    return {
        "median_output_kl": float(statistics.median(kl)),
        "output_kl_q25": quantile(kl, 0.25),
        "output_kl_q75": quantile(kl, 0.75),
        "median_prediction_flip_rate": float(statistics.median(flips)),
        "prediction_flip_rate_q25": quantile(flips, 0.25),
        "prediction_flip_rate_q75": quantile(flips, 0.75),
        "num_candidates": len(candidate_rows),
    }


def monotone_log_fit(
    scales: Sequence[float],
    distances: Sequence[float],
) -> list[dict[str, float]]:
    """Fit a non-decreasing calibration curve with log-space PAVA."""

    if len(scales) != len(distances) or len(scales) < 2:
        raise ValueError("at least two aligned scale/distance pairs are required")
    pairs = sorted((float(scale), float(distance)) for scale, distance in zip(scales, distances))
    if any(scale <= 0.0 or distance <= 0.0 for scale, distance in pairs):
        raise ValueError("scales and distances must be positive")
    if len({scale for scale, _ in pairs}) != len(pairs):
        raise ValueError("scales must be unique")

    x = [math.log(scale) for scale, _ in pairs]
    y = [math.log(distance) for _, distance in pairs]
    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(y):
        blocks.append({"start": index, "stop": index + 1, "weight": 1, "mean": value})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            weight = int(left["weight"]) + int(right["weight"])
            mean = (
                int(left["weight"]) * float(left["mean"])
                + int(right["weight"]) * float(right["mean"])
            ) / weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "stop": int(right["stop"]),
                    "weight": weight,
                    "mean": mean,
                }
            )
    fitted = [0.0] * len(pairs)
    for block in blocks:
        for index in range(int(block["start"]), int(block["stop"])):
            fitted[index] = float(block["mean"])
    return [
        {
            "scale": scale,
            "raw_distance": distance,
            "fitted_distance": math.exp(fitted[index]),
            "log_scale": x[index],
            "log_fitted_distance": fitted[index],
        }
        for index, (scale, distance) in enumerate(pairs)
    ]


def _interpolation_bracket(values: Sequence[float], target: float) -> tuple[int, int]:
    tolerance = 1e-12 * max(1.0, abs(target))
    if target < values[0] - tolerance or target > values[-1] + tolerance:
        raise ValueError(
            f"target {target:g} lies outside [{values[0]:g}, {values[-1]:g}]"
        )
    if target <= values[0] + tolerance:
        return 0, 0
    if target >= values[-1] - tolerance:
        return len(values) - 1, len(values) - 1
    for upper in range(1, len(values)):
        if target <= values[upper] + tolerance:
            return upper - 1, upper
    raise AssertionError("interpolation bracket was not found")


def interpolate_distance_for_scale(
    fitted_rows: Sequence[Mapping[str, float]],
    scale: float,
) -> float:
    """Evaluate a monotone fitted curve by log--log interpolation."""

    rows = sorted(fitted_rows, key=lambda row: float(row["scale"]))
    scales = [float(row["scale"]) for row in rows]
    lower, upper = _interpolation_bracket(scales, float(scale))
    if lower == upper:
        return float(rows[lower]["fitted_distance"])
    log_scale = math.log(float(scale))
    x0, x1 = math.log(scales[lower]), math.log(scales[upper])
    y0 = math.log(float(rows[lower]["fitted_distance"]))
    y1 = math.log(float(rows[upper]["fitted_distance"]))
    fraction = (log_scale - x0) / (x1 - x0)
    return math.exp(y0 + fraction * (y1 - y0))


def interpolate_scale_for_distance(
    fitted_rows: Sequence[Mapping[str, float]],
    target_distance: float,
) -> float:
    """Invert a monotone fitted curve by deterministic log interpolation."""

    rows = sorted(fitted_rows, key=lambda row: float(row["scale"]))
    distances = [float(row["fitted_distance"]) for row in rows]
    lower, upper = _interpolation_bracket(distances, float(target_distance))
    if lower == upper:
        return float(rows[lower]["scale"])
    y0, y1 = math.log(distances[lower]), math.log(distances[upper])
    x0, x1 = math.log(float(rows[lower]["scale"])), math.log(float(rows[upper]["scale"]))
    if math.isclose(y0, y1, rel_tol=0.0, abs_tol=1e-15):
        return math.exp(0.5 * (x0 + x1))
    fraction = (math.log(float(target_distance)) - y0) / (y1 - y0)
    return math.exp(x0 + fraction * (x1 - x0))


def build_three_matched_targets(
    randopt_fit: Sequence[Mapping[str, float]],
    modular_fit: Sequence[Mapping[str, float]],
    *,
    reference_modular_scale: float,
) -> list[dict[str, float | str]]:
    """Construct low/reference/high targets within the common KL support."""

    rand_distances = [float(row["fitted_distance"]) for row in randopt_fit]
    modular_distances = [float(row["fitted_distance"]) for row in modular_fit]
    overlap_low = max(min(rand_distances), min(modular_distances))
    overlap_high = min(max(rand_distances), max(modular_distances))
    if not overlap_low < overlap_high:
        raise ValueError("RandOpt and Modular calibration curves have no KL overlap")
    reference_kl = interpolate_distance_for_scale(
        modular_fit, reference_modular_scale
    )
    if not overlap_low < reference_kl < overlap_high:
        raise ValueError(
            "the canonical Modular scale is not strictly inside the common KL range; "
            "expand one or both calibration grids"
        )
    low_kl = math.sqrt(overlap_low * reference_kl)
    high_kl = math.sqrt(reference_kl * overlap_high)
    targets = []
    for name, target_kl in (
        ("low", low_kl),
        ("reference", reference_kl),
        ("high", high_kl),
    ):
        modular_scale = (
            float(reference_modular_scale)
            if name == "reference"
            else interpolate_scale_for_distance(modular_fit, target_kl)
        )
        randopt_scale = interpolate_scale_for_distance(randopt_fit, target_kl)
        targets.append(
            {
                "name": name,
                "target_median_output_kl": float(target_kl),
                "randopt_scale": float(randopt_scale),
                "modular_scale": float(modular_scale),
                "randopt_fitted_kl": interpolate_distance_for_scale(
                    randopt_fit, randopt_scale
                ),
                "modular_fitted_kl": interpolate_distance_for_scale(
                    modular_fit, modular_scale
                ),
            }
        )
    return targets


__all__ = [
    "DISTANCE_MATCH_PROTOCOL",
    "build_three_matched_targets",
    "directional_kl_per_position",
    "interpolate_distance_for_scale",
    "interpolate_scale_for_distance",
    "monotone_log_fit",
    "prediction_flip_per_position",
    "quantile",
    "summarize_candidate_examples",
    "summarize_condition_candidates",
]
