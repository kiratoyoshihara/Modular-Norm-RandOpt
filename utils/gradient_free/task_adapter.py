"""Adapters from the repository's canonical tasks to ES-at-Scale.

Prompts are rendered once and passed to vLLM as token IDs.  This prevents the
external trainer from applying a second chat template or adding a second BOS.
The reward and correctness decisions delegate to the existing data handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from data_handlers import get_dataset_handler
from utils.official_prompt_protocol import (
    encode_rendered_prompts,
    prompt_manifest,
    render_prompts,
)

from .protocol import SplitSpec, sha256_file


@dataclass(frozen=True)
class PreparedTaskSplit(Sequence[Mapping[str, Any]]):
    task: str
    role: str
    source_path: str
    source_sha256: str
    rows: tuple[Mapping[str, Any], ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    prompt_metadata: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        row = self.rows[index]
        return {
            "prompt": {"prompt_token_ids": list(self.prompt_token_ids[index])},
            "target": row["ground_truth"],
        }

    @property
    def total_prompt_tokens(self) -> int:
        return sum(len(ids) for ids in self.prompt_token_ids)


def collate_es_batch(
    batch: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Any]]:
    return [item["prompt"] for item in batch], [item["target"] for item in batch]


def identity_template(prompt: Any) -> Any:
    """ES-at-Scale template hook for already-tokenized canonical prompts."""

    return prompt


@lru_cache(maxsize=None)
def _handler(task: str) -> Any:
    return get_dataset_handler(task)


def score_response(response: str, target: Any, *, task: str) -> tuple[str, float]:
    handler = _handler(task)
    reward = float(handler.compute_reward(response, target))
    label = "correct" if handler.is_answer_correct(response, target) else "incorrect"
    return label, reward


def make_reward_function(task: str) -> Callable[[str, Any], tuple[str, float]]:
    # functools.partial of a module-level function remains pickleable by the
    # multiprocessing pool used inside the official trainer.
    _handler(task)
    return partial(score_response, task=task)


def load_task_rows(
    task: str,
    path: str | Path,
    *,
    split: str,
    max_samples: int | None = None,
) -> list[Mapping[str, Any]]:
    rows = list(_handler(task).load_data(str(path), split=split, max_samples=None))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    return rows


def prepare_task_split(
    *,
    task: str,
    role: str,
    path: str | Path,
    tokenizer: Any,
    model_name: str,
    chat_template_date: str,
    max_samples: int | None = None,
    expected: SplitSpec | None = None,
) -> PreparedTaskSplit:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = sha256_file(source)
    if expected is not None and source_digest != expected.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source}: expected {expected.sha256}, got {source_digest}"
        )

    all_rows = load_task_rows(task, source, split=role, max_samples=None)
    if expected is not None and len(all_rows) != expected.num_examples:
        raise ValueError(
            f"Row-count mismatch for {source}: expected {expected.num_examples}, "
            f"got {len(all_rows)}"
        )
    rows = all_rows if max_samples is None else all_rows[:max_samples]
    if not rows:
        raise ValueError(f"No examples loaded from {source}")

    rendered = render_prompts(
        tokenizer,
        model_name,
        rows,
        chat_template_date=chat_template_date,
    )
    token_ids = encode_rendered_prompts(tokenizer, rendered)
    metadata = prompt_manifest(
        tokenizer,
        token_ids,
        chat_template_date=chat_template_date,
    )
    metadata = {
        **metadata,
        "task": task,
        "role": role,
        "source_path": str(source),
        "source_sha256": source_digest,
        "full_source_examples": len(all_rows),
        "selected_examples": len(rows),
    }
    return PreparedTaskSplit(
        task=task,
        role=role,
        source_path=str(source),
        source_sha256=source_digest,
        rows=tuple(rows),
        prompt_token_ids=tuple(tuple(int(token) for token in ids) for ids in token_ids),
        prompt_metadata=metadata,
    )


def iter_targets(split: PreparedTaskSplit) -> Iterator[Any]:
    for row in split.rows:
        yield row["ground_truth"]


__all__ = [
    "PreparedTaskSplit",
    "collate_es_batch",
    "identity_template",
    "iter_targets",
    "load_task_rows",
    "make_reward_function",
    "prepare_task_split",
    "score_response",
]
