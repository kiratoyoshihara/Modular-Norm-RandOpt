"""Shared helpers that reproduce the official RandOpt sampling protocol.

This module deliberately mirrors the implementation in
https://github.com/sunrainyg/RandOpt.  It is used by the functional-radius
diagnostic only; the population search keeps the original implementation in
``population_scaling.py`` and ``utils/worker_extn.py`` unchanged.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


CANDIDATE_SEED_SCHEME = "numpy-default-rng-choice-2^31-v1"
PARAMETER_NOISE_SCHEME = "official-randopt-parameter-reseed-v1"
WEIGHT_RESTORE_SCHEME = "official-randopt-regenerate-subtract-v1"


def build_candidate_seeds(global_seed: int, population_size: int) -> List[int]:
    """Return the exact ordered seed population used by RandOpt.

    The population size is intentionally part of NumPy's ``choice`` call.
    Calibration controls therefore generate the full downstream Nmax pool and
    take a predeclared prefix from it, preserving exact seed correspondence.
    """

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    rng = np.random.default_rng(seed=int(global_seed))
    return (
        rng.choice(2**31, size=int(population_size), replace=False)
        .astype(np.int64)
        .tolist()
    )


def sample_parameter_noise(
    shape,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    candidate_seed: int,
) -> torch.Tensor:
    """Sample one tensor exactly as official ``worker_extn.py`` does.

    A fresh generator is initialized from the candidate seed for every
    physical parameter tensor.  The parameter name is intentionally not part
    of the seed.
    """

    generator = torch.Generator(device=device)
    generator.manual_seed(int(candidate_seed))
    return torch.randn(
        tuple(int(size) for size in shape),
        dtype=dtype,
        device=device,
        generator=generator,
    )


__all__ = [
    "CANDIDATE_SEED_SCHEME",
    "PARAMETER_NOISE_SCHEME",
    "WEIGHT_RESTORE_SCHEME",
    "build_candidate_seeds",
    "sample_parameter_noise",
]
