"""Portable model, data, and result handling for the iterative baselines."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_key(result):
    if not all(math.isfinite(result[k]) for k in ("accuracy", "mean_reward")):
        raise ValueError("Non-finite development score")
    return result["accuracy"], result["mean_reward"], -result["step"]


def development_indices(pool, training, test, count=500):
    key = lambda row: fingerprint(row["messages"])
    excluded = {key(row) for row in [*training, *test]}
    candidates, seen = [], set()
    for index, row in enumerate(pool):
        identity = key(row)
        if identity not in excluded and identity not in seen:
            candidates.append((identity, index))
            seen.add(identity)
    if len(candidates) < count:
        raise ValueError("Insufficient disjoint GSM8K development examples")
    return [index for _, index in sorted(candidates)[:count]]


def load_model(config, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
        dtype=dtype, attn_implementation="sdpa").to("cuda:0")
    model.eval()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("Embedding/head tie was lost")
    torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def load_split(task, role, tokenizer, config, data_root):
    from data_handlers import get_dataset_handler
    from utils.official_prompt_protocol import render_prompts, encode_rendered_prompts
    root = Path(data_root)
    handler = get_dataset_handler(task)
    if task == "countdown":
        filename = "validation" if role == "development" else role
        rows = handler.load_data(str(root / f"countdown/countdown_{filename}.json"))
    else:
        def load(name):
            return handler.load_data(str(root / "gsm8k" / name))
        if role == "development":
            pool = load("train.parquet")
            indices = development_indices(pool, load("train_200.parquet"), load("test.parquet"))
            rows = [pool[i] for i in indices]
        else:
            rows = load("train_200.parquet" if role == "train" else "test.parquet")
    expected = 200 if role == "train" else 500 if role == "development" else 1500 if task == "countdown" else 1319
    if len(rows) != expected:
        raise ValueError(f"{task}/{role}: expected {expected} rows, got {len(rows)}")
    if config.get("smoke"):
        rows = rows[:2]
    ids = encode_rendered_prompts(tokenizer, render_prompts(
        tokenizer, config["model"], rows, chat_template_date=config["chat_template_date"]))
    return handler, rows, ids


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)
