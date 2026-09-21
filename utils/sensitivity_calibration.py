"""Local-Jacobian sensitivity estimation utilities.

The functions in this module do not depend on Transformers and are unit-testable
with ordinary PyTorch modules.  The command-line calibration script uses them on
captured Qwen activations with the model frozen in eval mode.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch

_EPS = 1e-12


def _tensor_norm(tensor: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor.float()).clamp_min(eps)


def _random_unit_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    vector = torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    return vector / _tensor_norm(vector).to(vector.dtype)


def first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(output, dict):
        for item in output.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Function output contains no tensor: {type(output)!r}")


def estimate_operator_norm_power(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    *,
    power_iterations: int = 3,
    seed: int = 0,
    eps: float = _EPS,
) -> float:
    """Estimate ||J(point)||_2 by power iteration on J^T J using jvp/vjp."""

    if power_iterations < 1:
        raise ValueError("power_iterations must be >= 1")
    if not hasattr(torch, "func"):
        raise RuntimeError("Sensitivity calibration requires torch.func")
    x = point.detach()
    vector = _random_unit_like(x, seed)

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        return first_tensor(function(value))

    for _ in range(power_iterations):
        _, jv = torch.func.jvp(wrapped, (x,), (vector,))
        jv_norm = _tensor_norm(jv, eps)
        if float(jv_norm.item()) <= eps:
            return eps
        cotangent = jv / jv_norm.to(jv.dtype)
        _, vjp_fn = torch.func.vjp(wrapped, x)
        jt_u = vjp_fn(cotangent)[0]
        jt_norm = _tensor_norm(jt_u, eps)
        if float(jt_norm.item()) <= eps:
            return eps
        vector = jt_u / jt_norm.to(jt_u.dtype)

    _, jv = torch.func.jvp(wrapped, (x,), (vector,))
    sigma = _tensor_norm(jv, eps).item()
    return max(float(sigma), eps)


def estimate_operator_norm_directional(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    *,
    directions: int = 3,
    seed: int = 0,
    eps: float = _EPS,
) -> float:
    """Return the largest random-direction Jacobian gain."""

    if directions < 1:
        raise ValueError("directions must be >= 1")
    x = point.detach()

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        return first_tensor(function(value))

    gains = []
    for index in range(directions):
        vector = _random_unit_like(x, seed + index)
        _, jv = torch.func.jvp(wrapped, (x,), (vector,))
        gains.append(float(_tensor_norm(jv, eps).item()))
    return max(max(gains), eps)


def estimate_operator_norm(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    *,
    estimator: str = "power",
    power_iterations: int = 3,
    directions: int = 3,
    seed: int = 0,
) -> float:
    if estimator == "power":
        return estimate_operator_norm_power(
            function,
            point,
            power_iterations=power_iterations,
            seed=seed,
        )
    if estimator == "directional":
        return estimate_operator_norm_directional(
            function,
            point,
            directions=directions,
            seed=seed,
        )
    raise ValueError("estimator must be 'power' or 'directional'")


def aggregate_log_quantile(values: Sequence[float], quantile: float = 0.90) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty sensitivity sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    tensor = torch.tensor([math.log(max(float(value), _EPS)) for value in values])
    result = torch.quantile(tensor, float(quantile)).item()
    return float(math.exp(result))


def summarize_values(values: Iterable[float]) -> dict:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "median": None, "max": None}
    tensor = torch.tensor(rows, dtype=torch.float64)
    return {
        "count": len(rows),
        "min": min(rows),
        "median": float(torch.median(tensor).item()),
        "max": max(rows),
    }
