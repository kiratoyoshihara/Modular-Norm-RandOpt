import pytest

from utils.gradient_free.budget import BudgetLedger
from utils.gradient_free.protocol import (
    candidate_evaluations,
    iterations_for_candidate_budget,
    model_prompt_evaluations,
)


def test_es_budget_matched_checkpoint_is_exact():
    assert candidate_evaluations(30, 10) == 300
    assert model_prompt_evaluations(30, 10, 200) == 60_000
    assert iterations_for_candidate_budget(300, 30) == 10
    assert iterations_for_candidate_budget(900, 30) == 30
    assert iterations_for_candidate_budget(3000, 30) == 100


def test_budget_ledger_separates_search_and_evaluation():
    ledger = BudgetLedger(population_size=30, prompts_per_candidate=200)
    for _ in range(10):
        ledger.begin_iteration()
        ledger.add_search_tokens(prompt_tokens=12_000, completion_tokens=20_000)
        ledger.complete_iteration()
    ledger.add_evaluation(
        num_prompts=500,
        prompt_tokens=25_000,
        completion_tokens=80_000,
    )
    ledger.assert_consistent()
    result = ledger.snapshot()

    assert result["completed_candidate_evaluations"] == 300
    assert result["completed_model_prompt_evaluations"] == 60_000
    assert result["evaluation_model_prompt_evaluations"] == 500
    assert result["search_completion_tokens"] == 200_000
    assert result["evaluation_completion_tokens"] == 80_000


def test_failed_iteration_counts_attempted_but_not_completed_budget():
    ledger = BudgetLedger(population_size=30, prompts_per_candidate=200)
    ledger.begin_iteration()
    ledger.assert_consistent()

    assert ledger.attempted_candidate_evaluations == 30
    assert ledger.completed_candidate_evaluations == 0
    assert ledger.attempted_model_prompt_evaluations == 6_000
    assert ledger.completed_model_prompt_evaluations == 0


def test_nondivisible_candidate_budget_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        iterations_for_candidate_budget(301, 30)
