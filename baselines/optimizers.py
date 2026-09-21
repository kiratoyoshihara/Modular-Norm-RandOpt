"""Task-adapted MeZO and ZO-Finetuner cores, with no GPU/runtime imports on CLI plan.

References are pinned in README.md. This is a port, not the upstream Trainer.
The ZO-Finetuner normalization (including embedding effective-size weighting)
and detached directional derivative / normalization in L2L follow the official
implementation. HF parameter names/order are used; tied weights occur once.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn


def parameter_layout(model):
    return [(name, list(p.shape)) for name, p in model.named_parameters() if p.requires_grad]


def noise_stream(parameters, seed):
    """One RNG stream per step, regenerated across +/- probes and the update."""
    generators = {}
    for name, parameter in parameters:
        device = parameter.device
        if device not in generators:
            generators[device] = torch.Generator(device=device).manual_seed(int(seed))
        yield name, parameter, torch.normal(
            0.0, 1.0, size=parameter.shape, device=device,
            dtype=parameter.dtype, generator=generators[device])


def apply_noise(parameters, seed, coefficient, scales=None):
    with torch.no_grad():
        for name, parameter, noise in noise_stream(parameters, seed):
            direction = noise if scales is None else noise * scales[name].detach()
            # Regenerate/add/subtract, not exact reset to a base snapshot.
            parameter.add_(coefficient * direction)


def two_point_step(parameters, seed, epsilon, lr, objective, scales=None):
    if epsilon <= 0 or lr <= 0:
        raise ValueError("epsilon and learning rate must be positive")
    apply_noise(parameters, seed, epsilon, scales)
    positive = float(objective())
    apply_noise(parameters, seed, -2.0 * epsilon, scales)
    negative = float(objective())
    apply_noise(parameters, seed, epsilon, scales)
    coefficient = (positive - negative) / (2.0 * epsilon)
    if not all(math.isfinite(x) for x in (positive, negative, coefficient)):
        raise FloatingPointError("Non-finite ZO objective/derivative; run is invalid")
    apply_noise(parameters, seed, -lr * coefficient, scales)
    return positive, negative, coefficient


class PerturbationGenerator(nn.Module):
    """One 5 -> 64 -> 1 Tanh network per unique trainable parameter tensor."""

    def __init__(self, parameters, logical_batch_size):
        super().__init__()
        self.names = [name for name, _ in parameters]
        self.shapes = [list(p.shape) for _, p in parameters]
        self.networks = nn.ModuleList([
            nn.Sequential(nn.Linear(5, 64), nn.Tanh(), nn.Linear(64, 1))
            for _ in parameters])
        # Official code weights embedding blocks by 15 * batch size * width.
        self.weights = [
            float(15 * logical_batch_size * p.shape[1])
            if "embed" in name and p.ndim == 2 else float(p.numel())
            for name, p in parameters]
        self.previous = {name: 1.0 for name in self.names}
        self.history = (0.0, 0.0)

    def scales(self, parameters, *, detach_normalization=False):
        raw = OrderedDict()
        for (name, p), net in zip(parameters, self.networks, strict=True):
            with torch.no_grad():
                features = torch.stack((p.mean(), p.var(), p.new_tensor(self.history[0]),
                                        p.new_tensor(self.history[1]), p.new_tensor(self.previous[name])))
            raw[name] = net(features).reshape(()).abs()
        first = next(iter(raw.values()))
        weights = first.new_tensor(self.weights, dtype=torch.float64)
        values = torch.stack(list(raw.values())).to(torch.float64)
        norm = (weights.sum() / (weights * values.square()).sum()).sqrt().to(first.dtype)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Invalid learned perturbation normalization")
        if detach_normalization:
            norm = norm.detach()
        self.previous = {name: float(value.detach()) for name, value in raw.items()}
        return OrderedDict((name, value * norm) for name, value in raw.items())

    def reset_history(self):
        self.previous = {name: 1.0 for name in self.names}
        self.history = (0.0, 0.0)

    def dynamics(self):
        return dict(previous=self.previous, history=list(self.history))

    def restore_dynamics(self, state):
        if set(state["previous"]) != set(self.names):
            raise ValueError("Generator history has a different parameter layout")
        self.previous = dict(state["previous"])
        self.history = tuple(state["history"])


def differentiable_meta_loss(model, parameters, seed, scales, derivative, lr, batch):
    """Official L2L surrogate: stop-gradient through probes and base weights.

    functional_call preserves tied embedding/head aliases without the upstream
    model-name-specific retie branches. Gradients reach the PertNN networks only.
    """
    updated = OrderedDict()
    for name, parameter, noise in noise_stream(parameters, seed):
        updated[name] = parameter.detach() - lr * float(derivative) * noise * scales[name]
    return torch.func.functional_call(model, updated, (), batch, tie_weights=True).loss
