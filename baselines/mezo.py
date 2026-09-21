"""MeZO's isotropic two-point SGD, preserving upstream operation ordering.

The generation mirror is shared with the task-adapted ZO baseline, not its
learned perturbation generator. No master-weight reset is performed.
"""
from __future__ import annotations

import math

import torch

from baselines.optimizers import noise_stream
from baselines.zo import Generation, weight_metrics


@torch.no_grad()
def perturb(parameters, seed, epsilon, factor):
    for _, parameter, noise in noise_stream(parameters, seed):
        parameter.add_((factor * noise) * epsilon)


@torch.no_grad()
def step(parameters, seed, epsilon, learning_rate, objective):
    if epsilon <= 0 or learning_rate <= 0:
        raise ValueError("Positive epsilon and learning rate required")
    perturb(parameters, seed, epsilon, 1.)
    positive = float(objective())
    perturb(parameters, seed, epsilon, -2.)
    negative = float(objective())
    perturb(parameters, seed, epsilon, 1.)
    derivative = (positive - negative) / (2. * epsilon)
    if not all(math.isfinite(x) for x in (positive, negative, derivative)):
        raise FloatingPointError("Non-finite MeZO objective/derivative")
    for _, parameter, noise in noise_stream(parameters, seed):
        # Do not combine learning_rate * derivative before tensor multiplication:
        # the official zero-weight-decay expression rounds the inner product first.
        parameter.sub_(learning_rate * (derivative * noise))
    return positive, negative, derivative


@torch.no_grad()
def replay_reference(parameters, seed, epsilon, learning_rate, positive, negative):
    """Independent literal upstream replay used ONLY on disposable pilot copies."""
    for factor in (1., -2., 1.):
        torch.manual_seed(seed)
        for _, parameter in parameters:
            noise = torch.normal(0., 1., size=parameter.shape,
                                 device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(parameter + factor * noise * epsilon)
    derivative = (positive - negative) / (2. * epsilon)
    torch.manual_seed(seed)
    for _, parameter in parameters:
        noise = torch.normal(0., 1., size=parameter.shape,
                             device=parameter.device, dtype=parameter.dtype)
        parameter.copy_(parameter - learning_rate * (derivative * noise))
