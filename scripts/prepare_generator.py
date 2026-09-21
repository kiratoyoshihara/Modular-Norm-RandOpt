#!/usr/bin/env python3
"""Reproduce ZO-Finetuner's supervised perturbation-generator preparation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import common as io


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/iterative.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = io.read(args.config)
    if args.dry_run:
        import json
        print(json.dumps(cfg["meta"], indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from baselines import preparation
    preparation.DATA_ROOT = args.data_root.resolve()
    path = preparation.DATA_ROOT / "countdown/countdown_train.json"
    contract = dict(config=cfg, training_sha256=io.digest(path))
    contract["fingerprint"] = io.fingerprint(contract)
    root = args.output.resolve()
    if root.exists() and not args.resume:
        p.error("Use a new output directory or explicitly --resume a saved preparation")
    if args.resume and io.read(root / "configuration.json") != contract:
        p.error("Configuration differs from the saved preparation")
    io.write(root / "configuration.json", contract)
    preparation.meta_train(root, contract, resume=args.resume)


if __name__ == "__main__":
    main()
