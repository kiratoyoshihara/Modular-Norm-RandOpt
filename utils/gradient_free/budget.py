"""Exact budget accounting for iterative gradient-free methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class BudgetLedger:
    population_size: int
    prompts_per_candidate: int
    attempted_iterations: int = 0
    completed_iterations: int = 0
    attempted_candidate_evaluations: int = 0
    completed_candidate_evaluations: int = 0
    attempted_model_prompt_evaluations: int = 0
    completed_model_prompt_evaluations: int = 0
    search_prompt_tokens: int = 0
    search_completion_tokens: int = 0
    evaluation_model_prompt_evaluations: int = 0
    evaluation_prompt_tokens: int = 0
    evaluation_completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.prompts_per_candidate <= 0:
            raise ValueError("prompts_per_candidate must be positive")

    def begin_iteration(self, num_prompts: int | None = None) -> None:
        prompts = self.prompts_per_candidate if num_prompts is None else int(num_prompts)
        if prompts <= 0:
            raise ValueError("num_prompts must be positive")
        self.attempted_iterations += 1
        self.attempted_candidate_evaluations += self.population_size
        self.attempted_model_prompt_evaluations += self.population_size * prompts

    def complete_iteration(self, num_prompts: int | None = None) -> None:
        prompts = self.prompts_per_candidate if num_prompts is None else int(num_prompts)
        if self.completed_iterations >= self.attempted_iterations:
            raise RuntimeError("Cannot complete an iteration that was not begun")
        self.completed_iterations += 1
        self.completed_candidate_evaluations += self.population_size
        self.completed_model_prompt_evaluations += self.population_size * prompts

    def add_search_tokens(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.search_prompt_tokens += int(prompt_tokens)
        self.search_completion_tokens += int(completion_tokens)

    def add_evaluation(
        self,
        *,
        num_prompts: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if min(num_prompts, prompt_tokens, completion_tokens) < 0:
            raise ValueError("evaluation counts must be non-negative")
        self.evaluation_model_prompt_evaluations += int(num_prompts)
        self.evaluation_prompt_tokens += int(prompt_tokens)
        self.evaluation_completion_tokens += int(completion_tokens)

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["search_total_tokens"] = (
            self.search_prompt_tokens + self.search_completion_tokens
        )
        payload["evaluation_total_tokens"] = (
            self.evaluation_prompt_tokens + self.evaluation_completion_tokens
        )
        return payload

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> "BudgetLedger":
        fields = cls.__dataclass_fields__
        kwargs = {key: int(payload[key]) for key in fields if key in payload}
        return cls(**kwargs)

    def assert_consistent(self) -> None:
        if self.completed_iterations > self.attempted_iterations:
            raise AssertionError("completed iterations exceed attempted iterations")
        expected_attempted = self.attempted_iterations * self.population_size
        expected_completed = self.completed_iterations * self.population_size
        if self.attempted_candidate_evaluations != expected_attempted:
            raise AssertionError(
                "attempted candidate evaluations do not equal iterations × population"
            )
        if self.completed_candidate_evaluations != expected_completed:
            raise AssertionError(
                "completed candidate evaluations do not equal iterations × population"
            )
        if self.attempted_model_prompt_evaluations != (
            expected_attempted * self.prompts_per_candidate
        ):
            raise AssertionError("attempted model–prompt budget is inconsistent")
        if self.completed_model_prompt_evaluations != (
            expected_completed * self.prompts_per_candidate
        ):
            raise AssertionError("completed model–prompt budget is inconsistent")


__all__ = ["BudgetLedger"]
