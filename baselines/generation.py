"""Generation-only vLLM adapter. All optimization remains on the HF model.

Pinned to a single-device, in-process vLLM worker. No RandOpt noise routines
are used: HF tensor boundaries, dtype, RNG order and update arithmetic stay
in optimizers.py. Copying current HF weights is synchronization, not a reset
of the optimized model to its original checkpoint.
"""
from __future__ import annotations

import math
import os
import time
from statistics import mean

import torch

from utils.hf_vllm_parameter_bridge import build_physical_parameter_bindings


class ParameterSource:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self, remove_duplicate=True):
        return iter(self.parameters)


class WeightMirror:
    """Strict, zero-transform HF -> vLLM slice copies; also testable on CPU."""

    def __init__(self, parameters, runtime_parameters):
        self.sources = dict(parameters)
        if len(self.sources) != len(parameters):
            raise ValueError("Duplicate HF parameter names")
        self.runtime = dict(runtime_parameters)
        rows = [dict(parameter_name=name, shape=list(p.shape)) for name, p in self.runtime.items()]
        self.bindings, _ = build_physical_parameter_bindings(ParameterSource(parameters), rows)
        self.pairs = []
        for binding in self.bindings:
            physical = self.runtime[binding.physical_name]
            for fragment in binding.fragments:
                target = physical if fragment.start is None else physical[fragment.start:fragment.stop]
                source = self.sources[fragment.name]
                if source.dtype != target.dtype or source.device != target.device:
                    raise ValueError(f"Dtype/device conversion forbidden: {fragment.name}")
                if source.data_ptr() == target.data_ptr():
                    raise ValueError("Inference mirror must not alias optimizer weights")
                self.pairs.append((fragment.name, source, target))

    @torch.no_grad()
    def copy(self, verify=False):
        for _, source, target in self.pairs:
            target.copy_(source)
        return self.check() if verify else dict(copied_tensors=len(self.pairs))

    @torch.no_grad()
    def check(self):
        mismatched = [name for name, source, target in self.pairs if not torch.equal(source, target)]
        if mismatched:
            raise RuntimeError(f"HF/vLLM weight mismatch: {mismatched[:10]}")
        return dict(exactly_equal=True, checked_tensors=len(self.pairs),
                    checked_elements=sum(p.numel() for _, p, _ in self.pairs))


class GenerationWorker:
    def zo_bind_sources(self, parameters, owner_pid):
        if os.getpid() != owner_pid:
            raise RuntimeError("Weight sharing requires an in-process worker")
        model = self.model_runner.model
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            raise RuntimeError("vLLM lost Qwen's embedding/output-head tie")
        self.zo_mirror = WeightMirror(parameters, list(model.named_parameters()))
        self.zo_weight_version = 0
        return dict(worker_pid=os.getpid(), tied_embedding=True,
                    hf_tensors=len(parameters), runtime_tensors=len(self.zo_mirror.runtime))

    def zo_sync_weights(self, verify=False):
        result = self.zo_mirror.copy(verify)
        self.zo_weight_version += 1
        torch.cuda.synchronize()
        return dict(**result, weight_version=self.zo_weight_version)

    def zo_check_weights(self):
        return self.zo_mirror.check()


def score_outputs(outputs, handler, tokenizer, rows, ids):
    if len(outputs) != len(rows) or len(rows) != len(ids):
        raise ValueError("Generation output count mismatch")
    items = []
    for index, (output, row, prompt) in enumerate(zip(outputs, rows, ids, strict=True)):
        if list(output.prompt_token_ids) != list(prompt) or len(output.outputs) != 1 or not output.finished:
            raise ValueError("Prompt/order/completion mismatch")
        suffix = list(output.outputs[0].token_ids)
        text = tokenizer.decode(suffix, skip_special_tokens=True)
        reward = float(handler.compute_reward(text, row["ground_truth"]))
        if not math.isfinite(reward):
            raise FloatingPointError("Non-finite reward")
        items.append(dict(index=index, response=text, reward=reward,
                          correct=bool(handler.is_answer_correct(text, row["ground_truth"])),
                          generated_tokens=len(suffix), prompt_tokens=len(prompt),
                          finish_reason=output.outputs[0].finish_reason))
    return items, dict(mean_reward=mean(x["reward"] for x in items),
                       accuracy=mean(x["correct"] for x in items), examples=len(items),
                       generated_tokens=sum(x["generated_tokens"] for x in items),
                       prompt_tokens=sum(x["prompt_tokens"] for x in items))


class VLLMGeneration:
    def __init__(self, model, tokenizer, config, snapshot):
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise RuntimeError("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vLLM")
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("Exactly one GPU must be visible")
        from vllm import LLM, SamplingParams
        self.tokenizer, self.config = tokenizer, config
        self.engine = LLM(
            model=snapshot, tokenizer=snapshot, dtype="bfloat16", seed=38,
            tensor_parallel_size=1, distributed_executor_backend="uni",
            worker_extension_cls="baselines.generation.GenerationWorker",
            enable_prefix_caching=False, enforce_eager=True, async_scheduling=False,
            max_model_len=4096, max_num_seqs=200, max_num_batched_tokens=8192,
            gpu_memory_utilization=0.5, kv_cache_memory_bytes=12 * 2**30,
            generation_config="vllm", disable_log_stats=True,
        )
        eos = model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [tokenizer.eos_token_id])
        self.sampling = SamplingParams(
            n=1, temperature=0.0, top_p=1.0, top_k=-1,
            max_tokens=config["max_new_tokens"], repetition_penalty=config["repetition_penalty"],
            presence_penalty=0.0, frequency_penalty=0.0,
            stop_token_ids=self.eos_ids, ignore_eos=False, skip_special_tokens=True,
        )
        self.binding = self.rpc("zo_bind_sources", list(model.named_parameters()), os.getpid())
        self.initial_audit = self.sync(verify=True)

    def rpc(self, name, *args):
        result = self.engine.collective_rpc(name, args=args)
        if len(result) != 1:
            raise RuntimeError("Expected one inference worker")
        return result[0]

    def sync(self, verify=False):
        # generate() drains all requests. Prefix caching is disabled, so old
        # weights cannot leak via reuse of another request's KV cache.
        if self.engine.llm_engine.has_unfinished_requests():
            raise RuntimeError("Cannot change weights while generation is pending")
        return self.rpc("zo_sync_weights", verify)

    def generate_scores(self, handler, rows, ids, *, verify=False):
        if any(len(prompt) + self.config["max_new_tokens"] > 4096 for prompt in ids):
            raise ValueError("Prompt exceeds configured context; no silent truncation")
        started = time.monotonic()
        synced = self.sync(verify)
        synced_at = time.monotonic()
        outputs = self.engine.generate([dict(prompt_token_ids=p) for p in ids],
                                       self.sampling, use_tqdm=False)
        torch.cuda.synchronize()
        generated_at = time.monotonic()
        items, summary = score_outputs(outputs, handler, self.tokenizer, rows, ids)
        summary.update(sync_seconds=synced_at - started, generation_seconds=generated_at - synced_at,
                       wall_seconds=time.monotonic() - started, weight_version=synced["weight_version"])
        return summary, items

    def close(self):
        self.engine.llm_engine.engine_core.shutdown()
