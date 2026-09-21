from __future__ import annotations

import math

import pytest
import torch

from utils.functional_displacement import (
    hidden_rms_displacement,
    radius_objective,
    select_radius,
    symmetric_kl_from_logits,
)


def test_symmetric_kl_is_zero_for_identical_logits():
    logits = torch.tensor([[1.0, -1.0, 0.5], [0.0, 2.0, -2.0]])
    assert symmetric_kl_from_logits(logits, logits) == pytest.approx(0.0, abs=1e-12)


def test_symmetric_kl_is_symmetric_and_respects_mask():
    first = torch.tensor([[1.0, 0.0], [100.0, -100.0]])
    second = torch.tensor([[0.0, 1.0], [-100.0, 100.0]])
    mask = torch.tensor([True, False])
    assert symmetric_kl_from_logits(first, second, mask) == pytest.approx(
        symmetric_kl_from_logits(second, first, mask)
    )


def test_hidden_displacement_matches_known_scaling():
    base = torch.ones(2, 3)
    metrics = hidden_rms_displacement(base, 1.5 * base)
    assert metrics["absolute_rms"] == pytest.approx(0.5)
    assert metrics["relative_rms"] == pytest.approx(0.5)


def test_radius_objective_and_tie_break_are_deterministic():
    assert radius_objective(2.0, 2.0, 3.0, 3.0) == pytest.approx(0.0)
    selected = select_radius(
        [
            {"radius": 0.16, "objective_j": 1.0},
            {"radius": 0.08, "objective_j": 1.0},
            {"radius": 0.32, "objective_j": 2.0},
        ]
    )
    assert math.isclose(float(selected["radius"]), 0.08)
