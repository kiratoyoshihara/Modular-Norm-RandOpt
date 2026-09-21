"""Dtype-aware inference and upstream-ordered ZO arithmetic; no exact reset."""
from __future__ import annotations

import math
import os
from collections import OrderedDict

import torch

from baselines.optimizers import noise_stream
from baselines.generation import VLLMGeneration


class Generation(VLLMGeneration):
    def __init__(self, model, tokenizer, config, snapshot, dtype):
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0" or not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("One visible GPU and an in-process inference worker are required")
        if dtype not in ("float16", "bfloat16") or next(model.parameters()).dtype != getattr(torch, dtype):
            raise ValueError("HF and vLLM precision must match exactly")
        from vllm import LLM, SamplingParams
        self.tokenizer, self.config = tokenizer, config
        self.engine = LLM(model=snapshot, tokenizer=snapshot, dtype=dtype, seed=38,
            tensor_parallel_size=1, distributed_executor_backend="uni",
            worker_extension_cls="baselines.generation.GenerationWorker",
            enable_prefix_caching=False, enforce_eager=True, async_scheduling=False,
            max_model_len=4096, max_num_seqs=200, max_num_batched_tokens=8192,
            gpu_memory_utilization=.5, kv_cache_memory_bytes=12 * 2**30,
            generation_config="vllm", disable_log_stats=True)
        eos = model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [tokenizer.eos_token_id])
        self.sampling = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1,
            max_tokens=config["max_new_tokens"], repetition_penalty=config["repetition_penalty"],
            presence_penalty=0., frequency_penalty=0., stop_token_ids=self.eos_ids,
            ignore_eos=False, skip_special_tokens=True)
        self.binding = self.rpc("zo_bind_sources", list(model.named_parameters()), os.getpid())
        self.initial_audit = self.sync(verify=True)


@torch.no_grad()
def amplitudes(generator, parameters):
    raw = OrderedDict()
    # Match upstream scalar accumulation and do not round the normalization
    # into the model dtype before multiplying by the generated direction.
    weighted_square, effective_size = 0., 0.
    for (name, p), network, weight in zip(parameters, generator.networks, generator.weights, strict=True):
        features = torch.stack((p.mean(), p.var(), p.new_tensor(generator.history[0]),
                                p.new_tensor(generator.history[1]), p.new_tensor(generator.previous[name])))
        raw[name] = network(features).reshape(()).abs()
        scalar = float(raw[name])
        effective_size += weight / 1e6
        weighted_square += (weight / 1e6) * scalar**2
    if weighted_square <= 0 or not math.isfinite(weighted_square):
        raise FloatingPointError("Invalid perturbation normalization")
    normalization = math.sqrt(effective_size) / math.sqrt(weighted_square)
    generator.previous = {name: float(value) for name, value in raw.items()}
    return raw, normalization


@torch.no_grad()
def perturb(parameters, seed, raw, normalization, epsilon, factor):
    for name, p, noise in noise_stream(parameters, seed):
        p.add_((noise * raw[name]) * (factor * epsilon * normalization))


@torch.no_grad()
def step(parameters, seed, raw, normalization, epsilon, lr, objective):
    if epsilon <= 0 or lr <= 0:
        raise ValueError("Positive epsilon and learning rate required")
    perturb(parameters, seed, raw, normalization, epsilon, 1.)
    positive = float(objective())
    perturb(parameters, seed, raw, normalization, epsilon, -2.)
    negative = float(objective())
    perturb(parameters, seed, raw, normalization, epsilon, 1.)
    projected = (positive - negative) / (epsilon * 20.)
    if not all(math.isfinite(x) for x in (positive, negative, projected)):
        raise FloatingPointError("Non-finite ZO losses/derivative")
    for name, p, noise in noise_stream(parameters, seed):
        estimate = (noise * raw[name]) * projected
        p.sub_(estimate * (lr * 10. * normalization))
    return positive, negative, projected


@torch.no_grad()
def weight_metrics(parameters, reference=None):
    diff2 = base2 = norm_diff2 = norm_base2 = 0.
    changed = elements = 0
    for name, parameter in parameters:
        flat = parameter.detach().reshape(-1)
        old = reference[name].reshape(-1) if reference is not None else None
        for start in range(0, flat.numel(), 2**20):
            value = flat[start:start + 2**20].float()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite model parameter: {name}")
            elements += value.numel()
            if old is not None:
                initial = old[start:start + 2**20].to(value.device, torch.float32)
                delta = value - initial
                d, b = float(delta.double().square().sum()), float(initial.double().square().sum())
                diff2 += d
                base2 += b
                changed += int(torch.count_nonzero(delta))
                if "norm" in name:
                    norm_diff2 += d
                    norm_base2 += b
    return dict(finite=True, relative_l2=math.sqrt(diff2 / base2) if base2 else 0.,
                norm_relative_l2=math.sqrt(norm_diff2 / norm_base2) if norm_base2 else 0.,
                changed_fraction=changed / elements if elements else 0., elements=elements)
