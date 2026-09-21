from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_handlers.math500 import MATH500Handler
from data_handlers.olympiadbench import OlympiadBenchHandler


def _write_processed_parquet(path, dataset: str) -> None:
    rows = []
    for index in range(3):
        rows.append({
            "data_source": dataset,
            "prompt": [{
                "role": "user",
                "content": f"Problem {index}. Put the answer in \\boxed{{}}.",
            }],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(index + 1)},
            "extra_info": {
                "index": index,
                "split": "test",
                "subject": "Algebra",
                "level": 3,
                "unique_id": f"math-{index}",
                "id": 100 + index,
                "answer_type": "Numerical",
            },
        })
    pd.DataFrame(rows).to_parquet(path)


def test_math500_loads_processed_parquet_without_rewriting_prompt(tmp_path):
    path = tmp_path / "math500.parquet"
    _write_processed_parquet(path, "HuggingFaceH4/MATH-500")

    rows = MATH500Handler().load_data(
        str(path), split="test", max_samples=1, start_index=1
    )

    assert len(rows) == 1
    assert rows[0]["messages"] == [{
        "role": "user",
        "content": "Problem 1. Put the answer in \\boxed{}.",
    }]
    assert rows[0]["ground_truth"] == "2"
    assert rows[0]["subject"] == "Algebra"
    assert rows[0]["level"] == 3
    assert rows[0]["unique_id"] == "math-1"


def test_olympiadbench_loads_processed_parquet_without_rewriting_prompt(tmp_path):
    path = tmp_path / "olympiadbench.parquet"
    _write_processed_parquet(path, "olympiadbench")

    rows = OlympiadBenchHandler().load_data(
        str(path), split="test", max_samples=1, start_index=2
    )

    assert len(rows) == 1
    assert rows[0]["messages"] == [{
        "role": "user",
        "content": "Problem 2. Put the answer in \\boxed{}.",
    }]
    assert rows[0]["ground_truth"] == "3"
    assert rows[0]["ground_truth_raw"] == "3"
    assert rows[0]["answer_type"] == "Numerical"
    assert rows[0]["id"] == 102


def test_processed_math_ground_truth_is_compatible_with_boxed_scorer(tmp_path):
    path = tmp_path / "math500.parquet"
    _write_processed_parquet(path, "HuggingFaceH4/MATH-500")
    handler = MATH500Handler()
    row = handler.load_data(str(path), max_samples=1)[0]

    assert handler.compute_reward(r"The answer is \\boxed{1}.", row["ground_truth"]) == 1.0
