#!/usr/bin/env python3
"""Run MN-RandOpt at N=3000/K=1, matching ES-at-Scale iteration 100."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.es_at_scale import run_mn_randopt_matched as matched  # noqa: E402


def main() -> int:
    matched.PROTOCOL = "mn-randopt-es-at-scale-final-v1"
    matched.ES_ITERATIONS = 100
    matched.POPULATION_SIZE = 3_000
    matched.ADDITIONAL_SOURCE_PATHS = (
        Path("scripts/baselines/es_at_scale/run_mn_randopt_es_final.py"),
    )
    return matched.main()


if __name__ == "__main__":
    raise SystemExit(main())
