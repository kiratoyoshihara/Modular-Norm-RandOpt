"""Supervised ZO generator preparation; optimizer arithmetic is unchanged."""
from __future__ import annotations
import math
import os
from pathlib import Path
import shutil
import time
from baselines.common import (now, digest, fingerprint, read, write, append,
                              load_model, save_checkpoint)
from baselines.common import load_split as _load_split

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
STOP = False

def load_split(task, role, tokenizer, config):
    return _load_split(task, role, tokenizer, config, DATA_ROOT)

def verify_contract(contract):
    body = dict(contract)
    expected = body.pop("fingerprint")
    if fingerprint(body) != expected:
        raise ValueError("Preparation configuration changed")
    if digest(DATA_ROOT / "countdown/countdown_train.json") != body["training_sha256"]:
        raise ValueError("Preparation training data changed")

def records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def meta_examples():
    from utils.reward_score.countdown import answer_reward_function
    raw = read(DATA_ROOT / "countdown/countdown_train.json")
    valid, excluded = [], []
    for index, row in enumerate(raw):
        solution = row.get("solution")
        response = f"<think></think>\n<answer>{solution}</answer>"
        if solution and answer_reward_function(response, row["numbers"], row["target"]) == 1:
            valid.append(dict(index=index, id=row["id"], response=response))
        else:
            excluded.append(row["id"])
    return valid, excluded

def require_checkpoint_space(directory):
    # Two full FP32 model+momentum checkpoints may coexist during atomic save.
    if shutil.disk_usage(directory).free < 25 * 2**30:
        raise RuntimeError("Less than 25 GiB disk space remains; stopping before new GPU work")

def attempt_time(directory):
    return sum(row["wall_seconds"] for row in records(directory / "attempts.jsonl"))

def begin_attempt(directory):
    marker = directory / "attempt_running.json"
    if marker.exists():
        raise RuntimeError("Prior attempt ended without a cost record (possibly SIGKILL); preserve it and use a new root")
    write(marker, dict(started_at=now(), pid=os.getpid()))

def validate_resume_ledger(directory, count, *, meta=False):
    if (directory / "inflight.json").exists():
        raise RuntimeError("Interrupted work was not committed; refusing hidden replay. Preserve this run and use a new output root")
    logged = records(directory / ("meta_steps.jsonl" if meta else "queries.jsonl"))
    paid = len(logged) if meta else sum(r["examples"] for r in logged)
    if paid != count:
        raise RuntimeError("Uncheckpointed work exists; refusing hidden replay. Preserve this run and use a new output root")

def generator_artifact(root, contract):
    path = root / "meta/generator.pt"
    summary_path = root / "meta/summary.json"
    if not path.is_file() or not summary_path.is_file():
        raise RuntimeError("Run meta-train first; a random/untrained generator is not a baseline")
    summary = read(summary_path)
    if summary["contract"] != contract["fingerprint"] or summary["artifact_sha256"] != digest(path):
        raise ValueError("Generator provenance mismatch")
    return path

def ce_batch(examples, tokenizer, device):
    import torch
    # Right padding for supervised learning; mask prompt and padding tokens.
    length = max(len(row["input_ids"]) for row in examples)
    ids, masks, labels = [], [], []
    for row in examples:
        padding = length - len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {name: torch.tensor(value, device=device) for name, value in
            (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}

def meta_train(root, contract, resume=False):
    import torch
    from baselines.optimizers import PerturbationGenerator, apply_noise, differentiable_meta_loss
    config, meta = contract["config"], contract["config"]["meta"]
    directory = root / "meta"
    if (directory / "summary.json").exists():
        generator_artifact(root, contract)
        return
    if directory.exists() and not resume:
        raise RuntimeError("Incomplete meta-training exists; inspect and use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    require_checkpoint_space(directory)
    begin_attempt(directory)
    started, success = time.monotonic(), False
    try:
        verify_contract(contract)
        torch.manual_seed(meta["seed"])
        model, tokenizer = load_model(config, torch.float32)
        model.config.use_cache = False
        parameters = list(model.named_parameters())
        original = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
        generator = PerturbationGenerator(parameters, meta["batch_size"]).to(model.device)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=meta["generator_lr"])
        trajectory = torch.optim.SGD(model.parameters(), lr=meta["trajectory_lr"], momentum=meta["trajectory_momentum"])
        _, _, train_ids = load_split("countdown", "train", tokenizer, config)
        valid, excluded = meta_examples()
        examples = []
        for row in valid:
            prompt = train_ids[row["index"]]
            target = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prompt) + len(target) > meta["max_sequence_length"]:
                raise ValueError("Meta example is too long; silent truncation is forbidden")
            examples.append(dict(input_ids=prompt + target, labels=[-100] * len(prompt) + target))
        write(directory / "training_data.json", dict(source="countdown/train", included_ids=[r["id"] for r in valid],
            excluded_ids=excluded, target_format="<think></think>\\n<answer>{saved_solution}</answer>",
            uses_supervised_targets=True, examples=len(examples)))
        batches_per_epoch = math.ceil(len(examples) / meta["batch_size"])
        cycle_steps = batches_per_epoch * meta["reset_every_epochs"]
        start_epoch, completed_epochs = 0, 0
        counters = dict(forward_example_evaluations=0, backward_example_evaluations=0, generator_update_steps=0)
        checkpoint = directory / "checkpoint.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["contract"] != contract["fingerprint"]:
                raise ValueError("Meta checkpoint contract mismatch")
            model.load_state_dict(state["model"])
            generator.load_state_dict(state["generator"])
            generator.restore_dynamics(state["dynamics"])
            generator_optimizer.load_state_dict(state["generator_optimizer"])
            trajectory.load_state_dict(state["trajectory_optimizer"])
            start_epoch = completed_epochs = state["completed_epochs"]
            counters = state["counters"]
        validate_resume_ledger(directory, completed_epochs * batches_per_epoch, meta=True)
        def checkpoint_epoch():
            save_checkpoint(checkpoint, dict(contract=contract["fingerprint"], model=model.state_dict(),
                generator=generator.state_dict(), dynamics=generator.dynamics(),
                generator_optimizer=generator_optimizer.state_dict(), trajectory_optimizer=trajectory.state_dict(),
                completed_epochs=completed_epochs, counters=counters))
        for epoch in range(start_epoch, meta["epochs"]):
            if STOP:
                checkpoint_epoch()
                raise InterruptedError("Meta-training stopped at an epoch boundary")
            verify_contract(contract)
            if epoch > 0 and epoch % meta["reset_every_epochs"] == 0:
                # Periodic trajectory restart is intrinsic to the published L2L method,
                # not a change to RandOpt's perturbation restoration.
                model.load_state_dict(original)
                trajectory = torch.optim.SGD(model.parameters(), lr=meta["trajectory_lr"], momentum=meta["trajectory_momentum"])
            order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(meta["seed"] + epoch)).tolist()
            for batch_index, offset in enumerate(range(0, len(order), meta["batch_size"])):
                write(directory / "inflight.json", dict(epoch=epoch + 1, batch=batch_index, at=now()))
                batch = ce_batch([examples[i] for i in order[offset:offset + meta["batch_size"]]], tokenizer, model.device)
                count = batch["input_ids"].shape[0]
                model.eval()
                scales = generator.scales(parameters, detach_normalization=True)
                seed = meta["seed"] * 1_000_003 + epoch * batches_per_epoch + batch_index
                apply_noise(parameters, seed, meta["epsilon"], scales)
                with torch.no_grad():
                    positive = float(model(**batch).loss)
                apply_noise(parameters, seed, -2 * meta["epsilon"], scales)
                with torch.no_grad():
                    negative = float(model(**batch).loss)
                apply_noise(parameters, seed, meta["epsilon"], scales)
                derivative = (positive - negative) / (2 * meta["epsilon"])
                generator_optimizer.zero_grad(set_to_none=True)
                loss = differentiable_meta_loss(model, parameters, seed, scales, derivative, meta["update_lr"], batch)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite L2L loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
                    raise FloatingPointError("Nonfinite PertNN gradient")
                counters["generator_update_steps"] += int(any(
                    p.grad is not None and torch.count_nonzero(p.grad).item() for p in generator.parameters()))
                generator_optimizer.step()
                generator.history = (positive, negative)
                trajectory.zero_grad(set_to_none=True)
                model.train()
                trajectory_loss = model(**batch).loss
                if not torch.isfinite(trajectory_loss).item():
                    raise FloatingPointError("Nonfinite trajectory loss")
                trajectory_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), meta["trajectory_grad_clip"], error_if_nonfinite=True)
                cycle_step = (epoch % meta["reset_every_epochs"]) * batches_per_epoch + batch_index
                trajectory.param_groups[0]["lr"] = meta["trajectory_lr"] * max(0, 1 - cycle_step / cycle_steps)
                trajectory.step()
                trajectory.zero_grad(set_to_none=True)
                counters["forward_example_evaluations"] += 4 * count
                counters["backward_example_evaluations"] += 2 * count
                append(directory / "meta_steps.jsonl", dict(epoch=epoch + 1, batch=batch_index,
                    meta_loss=float(loss.detach()), trajectory_loss=float(trajectory_loss.detach()), derivative=derivative,
                    **counters, at=now()))
                (directory / "inflight.json").unlink()
                del loss, trajectory_loss, scales
            completed_epochs = epoch + 1
            checkpoint_epoch()
            print(f"meta-training: {completed_epochs}/{meta['epochs']} epochs", flush=True)
        if counters["generator_update_steps"] == 0:
            raise RuntimeError("Generator received zero gradients throughout; refusing to publish an untrained artifact")
        artifact = directory / "generator.pt"
        save_checkpoint(artifact, dict(layout=[[name, list(p.shape)] for name, p in parameters],
            networks={name: value.detach().cpu() for name, value in generator.state_dict().items()},
            contract=contract["fingerprint"], meta_seed=meta["seed"]))
        write(directory / "summary.json", dict(status="complete", contract=contract["fingerprint"],
            artifact_sha256=digest(artifact), examples=len(examples), epochs=meta["epochs"],
            **counters, wall_seconds_including_prior_attempts=attempt_time(directory) + time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            downstream_starts_from="original pretrained checkpoint, NOT this SGD trajectory",
            history_policy="reset before each downstream run", finished_at=now()))
        success = True
    finally:
        append(directory / "attempts.jsonl", dict(at=now(), wall_seconds=time.monotonic() - started, successful=success))
        (directory / "attempt_running.json").unlink()
