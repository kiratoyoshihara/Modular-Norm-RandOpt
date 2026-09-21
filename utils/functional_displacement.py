"""Pure metric helpers for accuracy-free functional-radius matching."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch


def _flatten_valid_rows(
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("expected at least [positions, features]")
    rows = tensor.reshape(-1, tensor.shape[-1])
    if mask is None:
        return rows
    valid = mask.reshape(-1).to(dtype=torch.bool, device=tensor.device)
    if valid.numel() != rows.shape[0]:
        raise ValueError("mask does not match the tensor's position dimensions")
    return rows[valid]


def symmetric_kl_from_logits(
    base_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    position_chunk_size: int = 16,
) -> float:
    """Mean symmetric KL across non-masked teacher-forced positions."""

    if base_logits.shape != perturbed_logits.shape:
        raise ValueError("base and perturbed logits must have identical shapes")
    if position_chunk_size < 1:
        raise ValueError("position_chunk_size must be positive")
    base_rows = _flatten_valid_rows(base_logits, mask)
    perturbed_rows = _flatten_valid_rows(perturbed_logits, mask)
    if base_rows.shape[0] == 0:
        raise ValueError("at least one teacher-forced position is required")

    total = 0.0
    count = 0
    for start in range(0, base_rows.shape[0], position_chunk_size):
        stop = min(start + position_chunk_size, base_rows.shape[0])
        base = base_rows[start:stop].float()
        perturbed = perturbed_rows[start:stop].float()
        log_p = torch.log_softmax(base, dim=-1)
        log_q = torch.log_softmax(perturbed, dim=-1)
        p = log_p.exp()
        q = log_q.exp()
        per_position = 0.5 * (
            (p * (log_p - log_q)).sum(dim=-1)
            + (q * (log_q - log_p)).sum(dim=-1)
        )
        total += float(per_position.double().sum().item())
        count += int(per_position.numel())
    return total / count


def hidden_rms_displacement(
    base_hidden: torch.Tensor,
    perturbed_hidden: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Mapping[str, float]:
    """Absolute and relative RMS displacement over valid hidden coordinates."""

    if base_hidden.shape != perturbed_hidden.shape:
        raise ValueError("base and perturbed hidden states must have identical shapes")
    base_rows = _flatten_valid_rows(base_hidden, mask).double()
    perturbed_rows = _flatten_valid_rows(perturbed_hidden, mask).double()
    if base_rows.numel() == 0:
        raise ValueError("at least one hidden coordinate is required")
    delta = perturbed_rows - base_rows
    absolute = math.sqrt(float(delta.square().mean().item()))
    base_rms = math.sqrt(float(base_rows.square().mean().item()))
    relative = absolute / max(base_rms, 1e-30)
    return {
        "absolute_rms": absolute,
        "base_rms": base_rms,
        "relative_rms": relative,
    }


def aggregate_candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Arithmetic candidate mean, fixed before inspecting downstream accuracy."""

    if not rows:
        raise ValueError("candidate metric rows must not be empty")
    keys = ("symmetric_kl", "hidden_absolute_rms", "hidden_relative_rms")
    aggregate = {
        key: sum(float(row[key]) for row in rows) / len(rows) for key in keys
    }
    layer_ids = sorted(
        {
            str(layer_id)
            for row in rows
            for layer_id in row.get("layerwise_hidden_relative_rms", {})
        },
        key=lambda value: int(value),
    )
    aggregate["layerwise_hidden_relative_rms"] = {
        layer_id: sum(
            float(row["layerwise_hidden_relative_rms"][layer_id]) for row in rows
        )
        / len(rows)
        for layer_id in layer_ids
    }
    aggregate["num_candidates"] = len(rows)
    return aggregate


def radius_objective(
    modular_kl: float,
    isotropic_kl: float,
    modular_hidden: float,
    isotropic_hidden: float,
) -> float:
    values = (modular_kl, isotropic_kl, modular_hidden, isotropic_hidden)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("KL and hidden displacement values must be finite and positive")
    return math.log(modular_kl / isotropic_kl) ** 2 + math.log(
        modular_hidden / isotropic_hidden
    ) ** 2


def select_radius(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select minimum J, breaking exact ties toward the smaller radius."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("radius rows must not be empty")
    return min(materialized, key=lambda row: (float(row["objective_j"]), float(row["radius"])))


__all__ = [
    "aggregate_candidate_metrics",
    "hidden_rms_displacement",
    "radius_objective",
    "select_radius",
    "symmetric_kl_from_logits",
]
