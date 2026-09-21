from __future__ import annotations

import pytest
import torch

from utils.hf_vllm_parameter_bridge import (
    apply_physical_perturbation,
    build_physical_parameter_bindings,
    snapshot_parameters,
)
from utils.official_randopt_protocol import sample_parameter_noise


class _SeparateProjectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 2, bias=False)
        self.v_proj = torch.nn.Linear(4, 2, bias=False)
        self.gate_proj = torch.nn.Linear(4, 3, bias=False)
        self.up_proj = torch.nn.Linear(4, 3, bias=False)


def _rows():
    return [
        {
            "parameter_name": "qkv_proj.weight",
            "shape": [8, 4],
            "role": "qkv",
            "multiplicity": 3.0,
            "scale": 1.0,
        },
        {
            "parameter_name": "gate_up_proj.weight",
            "shape": [6, 4],
            "role": "gate_up",
            "multiplicity": 2.0,
            "scale": 1.0,
        },
    ]


def test_official_fused_noise_is_sliced_into_hf_parameters():
    model = _SeparateProjectionModel().to(dtype=torch.float32)
    bindings, parameters = build_physical_parameter_bindings(model, _rows())
    snapshot = snapshot_parameters(parameters)
    radius = 0.01
    seed = 123
    apply_physical_perturbation(
        bindings=bindings,
        parameters=parameters,
        candidate_seed=seed,
        radius=radius,
        method="isotropic",
        power_iterations=2,
    )
    qkv = sample_parameter_noise(
        (8, 4), dtype=torch.float32, device=torch.device("cpu"), candidate_seed=seed
    )
    expected = {
        "q_proj.weight": qkv[:4],
        "k_proj.weight": qkv[4:6],
        "v_proj.weight": qkv[6:],
    }
    for name, fragment in expected.items():
        assert torch.allclose(
            parameters[name], snapshot[name] + radius * fragment, rtol=1e-7, atol=1e-8
        )


def test_official_bridge_restores_by_regenerating_and_subtracting():
    model = _SeparateProjectionModel().to(dtype=torch.float32)
    bindings, parameters = build_physical_parameter_bindings(model, _rows())
    snapshot = snapshot_parameters(parameters)
    kwargs = dict(
        bindings=bindings,
        parameters=parameters,
        candidate_seed=44,
        radius=0.02,
        method="isotropic",
        power_iterations=2,
    )
    apply_physical_perturbation(**kwargs, restore=False)
    apply_physical_perturbation(**kwargs, restore=True)
    for name, parameter in parameters.items():
        assert torch.allclose(parameter, snapshot[name], rtol=0.0, atol=2e-7)


def test_bridge_rejects_incomplete_runtime_layout():
    with pytest.raises(ValueError, match="absent from the vLLM"):
        build_physical_parameter_bindings(_SeparateProjectionModel(), _rows()[:1])
