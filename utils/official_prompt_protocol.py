"""Shared prompt construction for official-RandOpt model transfer.

Chat templates are rendered exactly once with a fixed date, then tokenized
without adding special tokens a second time.  The resulting token ids are used
by calibration, functional-displacement selection, and population evaluation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


PROMPT_TOKENIZATION_SCHEME = "chat-template-single-tokenization-fixed-date-v1"
DEFAULT_CHAT_TEMPLATE_DATE = "11 Aug 2026"


def render_prompts(
    tokenizer: Any,
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    chat_template_date: str,
) -> list[str]:
    if not chat_template_date:
        raise ValueError("chat_template_date must be fixed and non-empty")
    is_chat_model = any(
        marker in model_name.lower() for marker in ("instruct", "chat", "-it")
    )
    prompts: list[str] = []
    for row in rows:
        messages = row["messages"]
        if is_chat_model and tokenizer.chat_template:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    date_string=chat_template_date,
                )
            )
        else:
            prompts.append("\n".join(message["content"] for message in messages) + "\n")
    return prompts


def encode_rendered_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int | None = None,
) -> list[list[int]]:
    token_ids: list[list[int]] = []
    for prompt in prompts:
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if max_length is not None and len(ids) > max_length:
            raise ValueError(
                f"Rendered prompt has {len(ids)} tokens, exceeding max_length={max_length}; "
                "silent truncation is forbidden"
            )
        token_ids.append(ids)
    assert_no_double_bos(tokenizer, token_ids)
    return token_ids


def assert_no_double_bos(tokenizer: Any, token_ids: Sequence[Sequence[int]]) -> None:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        return
    bad = [
        index
        for index, ids in enumerate(token_ids)
        if len(ids) >= 2 and ids[0] == bos_token_id and ids[1] == bos_token_id
    ]
    if bad:
        raise ValueError(
            f"Double BOS detected in {len(bad)} prompts (first indices: {bad[:5]})"
        )


def prompt_manifest(
    tokenizer: Any,
    token_ids: Sequence[Sequence[int]],
    *,
    chat_template_date: str,
) -> dict[str, Any]:
    normalized = [[int(token) for token in ids] for ids in token_ids]
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    double_bos_count = sum(
        len(ids) >= 2 and bos_token_id is not None
        and ids[0] == bos_token_id and ids[1] == bos_token_id
        for ids in normalized
    )
    return {
        "scheme": PROMPT_TOKENIZATION_SCHEME,
        "tokenization_scheme": PROMPT_TOKENIZATION_SCHEME,
        "chat_template_date": chat_template_date,
        "add_special_tokens_after_render": False,
        "num_prompts": len(normalized),
        "num_examples": len(normalized),
        "total_tokens": sum(len(ids) for ids in normalized),
        "min_tokens": min((len(ids) for ids in normalized), default=0),
        "max_tokens": max((len(ids) for ids in normalized), default=0),
        "token_ids_sha256": hashlib.sha256(payload).hexdigest(),
        "bos_token_id": bos_token_id,
        "double_bos_prompt_count": int(double_bos_count),
        "leading_token_ids": [ids[:4] for ids in normalized[:3]],
    }


__all__ = [
    "DEFAULT_CHAT_TEMPLATE_DATE",
    "PROMPT_TOKENIZATION_SCHEME",
    "assert_no_double_bos",
    "encode_rendered_prompts",
    "prompt_manifest",
    "render_prompts",
]
