#!/usr/bin/env python3
"""Run the existing J(r) measurement with OLMo's padded-vocabulary bridge.

Only bindings for embedding/head padding and source provenance are extended.
No shared source, model/config tensors, or experimental settings are modified.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import measure_functional_displacement as measurement
from utils.olmo3_padded_vocab_bridge import build_physical_parameter_bindings, source_manifest


def main(argv=None):
    args = measurement.parse_args(argv)
    if args.model_name != "allenai/Olmo-3-7B-Instruct":
        raise ValueError("This launch-local compatibility entry point is for OLMo-3-7B-Instruct only")
    measurement.build_physical_parameter_bindings = build_physical_parameter_bindings
    measurement.source_manifest = source_manifest
    return measurement.run(args)


if __name__ == "__main__":
    main()
