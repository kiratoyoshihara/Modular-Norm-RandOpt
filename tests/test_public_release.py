from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.summarize_run import metric
from baselines.common import checkpoint_key, development_indices
from baselines.mezo import step, replay_reference
from scripts.run_experiment import parser, build_command
from scripts.run_baseline import configuration, parser as baseline_parser
from utils.masked_worker import mask_scales
from utils.official_randopt_protocol import build_candidate_seeds, sample_parameter_noise

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ("mn", "randopt", "attention-only", "mlp-only", "rmsnorm-weights-only", "rmsnorm-scale-correction", "frobenius"))
def test_portable_population_commands(method, tmp_path):
    args = parser().parse_args(["--method", method, "--gpu", "3", "--output", str(tmp_path / "run"), "--dry-run"])
    command, options = build_command(args)
    assert Path(command[1]).is_file()
    assert options["cuda_devices"] == 3
    assert options["population_size"] == 100
    assert options["max_tokens"] == 1024
    assert not (tmp_path / "run").exists()
    if method in ("attention-only", "mlp-only"):
        assert options["radius"] == .16
        assert json.loads(options["mass_config"])["attention"] == .5
        assert options["parameter_mask"] == method.removesuffix("-only")


def test_mask_is_applied_to_the_engine_not_the_tokenizer():
    tree = ast.parse((ROOT / "population_scaling_ablation.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    launch = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "launch_engines")
    assert "worker_extension_cls" in {key.arg for key in launch.keywords}
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            assert "worker_extension_cls" not in {key.arg for key in node.keywords}


@pytest.mark.parametrize("family", ("attention", "mlp"))
def test_mask_retains_full_scales_exactly(family):
    scales = {"model.layers.0.self_attn.qkv_proj.weight": 7.5,
              "model.layers.0.mlp.gate_up_proj.weight": 2.25,
              "model.layers.0.input_layernorm.weight": 4.0}
    masked = mask_scales(scales, family)
    keep = next(name for name in scales if ("self_attn" if family == "attention" else "mlp") in name)
    assert masked[keep] == scales[keep]
    assert all(value == float("inf") for name, value in masked.items() if name != keep)
    assert all(np.isfinite(value) for value in scales.values())


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_mezo_update_is_bitwise_identical_to_reference(dtype):
    a = [("w", torch.nn.Parameter(torch.tensor([.1, .2, -.3], dtype=dtype)))]
    b = [(name, torch.nn.Parameter(p.detach().clone())) for name, p in a]
    losses = iter((.3, .2))
    step(a, 42, .001, 1e-6, lambda: next(losses))
    replay_reference(b, 42, .001, 1e-6, .3, .2)
    assert torch.equal(a[0][1], b[0][1])


def test_candidate_noise_protocol_preserved():
    expected = np.random.default_rng(42).choice(2**31, size=100, replace=False).tolist()
    assert build_candidate_seeds(42, 100) == expected
    generator = torch.Generator().manual_seed(10)
    expected_noise = torch.randn((3, 5), generator=generator, dtype=torch.bfloat16)
    assert torch.equal(sample_parameter_noise((3, 5), dtype=torch.bfloat16,
        device=torch.device("cpu"), candidate_seed=10), expected_noise)


def test_development_selection_is_question_only_and_disjoint():
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(8)]
    ids = development_indices(rows, rows[:1], rows[1:2], count=4)
    assert len(ids) == 4 and not set(ids).intersection((0, 1))
    assert ids == development_indices(rows, rows[:1], rows[1:2], count=4)
    assert checkpoint_key(dict(accuracy=.5, mean_reward=.5, step=0)) > checkpoint_key(dict(accuracy=.5, mean_reward=.5, step=100))


def test_paper_metrics():
    name, score = metric("rocstories", ["ABCDE", "ACBDE"], [{"gold_labels": list("ABCDE")}] * 2, None)
    assert name == "exact_match" and score == .5
    name, score = metric("uspto50k", ["1", "1", "1"], ["1", "1", "2"], None)
    assert name == "balanced_accuracy" and score == .5


def test_figure_means_and_sample_sds():
    frame = pd.read_csv(ROOT / "results/figure2_population_scaling_data.csv")
    values = frame[[f"accuracy_seed_{seed}_percent" for seed in (42, 43, 44)]].to_numpy()
    np.testing.assert_allclose(values.mean(axis=1), frame.accuracy_mean_percent, atol=1e-12)
    np.testing.assert_allclose(values.std(axis=1, ddof=1), frame.accuracy_sample_sd_percent, atol=1e-12)
    ratios = pd.read_csv(ROOT / "results/figure4_required_population_ratio.csv")
    values = ratios[[f"ratio_seed_{seed}" for seed in (42, 43, 44)]].to_numpy()
    np.testing.assert_allclose(values.mean(axis=1), ratios.tail_implied_candidate_reduction_ratio, equal_nan=True)
    np.testing.assert_allclose(values.std(axis=1, ddof=1), ratios.ratio_sample_sd, equal_nan=True)


def test_generator_checksum():
    meta = json.loads((ROOT / "artifacts/zo_generator.json").read_text())
    assert hashlib.sha256((ROOT / "artifacts/zo_generator.pt").read_bytes()).hexdigest() == meta["sha256"]


@pytest.mark.parametrize("value", ("nan", "inf", "0", "-1"))
def test_baseline_rejects_invalid_learning_rate(value, tmp_path):
    args = baseline_parser().parse_args(["--method", "mezo", "--task", "gsm8k",
        "--phase", "smoke", f"--learning-rate={value}", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="finite and positive"):
        configuration(args)


def test_source_manifests_are_self_contained():
    from utils.official_randopt_provenance import source_manifest
    from utils.olmo3_padded_vocab_bridge import source_manifest as olmo_manifest
    for provider in (source_manifest, olmo_manifest):
        result = provider(ROOT)
        assert len(result["combined_sha256"]) == 64
        assert all((ROOT / name).is_file() for name in result["files"])


def test_es_comparison_dry_run_keeps_the_full_seed_pool(tmp_path):
    output = tmp_path / "search"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_es_comparison.py"),
        "--phase", "search", "--task", "countdown", "--seed", "43", "--gpu", "2",
        "--output", str(output), "--dry-run"], capture_output=True, text=True, check=True)
    assert "--population_size 3000" in result.stdout
    assert "--population_prefixes 3000" in result.stdout
    assert "--global_seed 43" in result.stdout
    assert "--cuda_devices 2" in result.stdout
    assert not output.exists()


def test_no_private_paths_or_extra_markdown():
    assert [p.relative_to(ROOT).as_posix() for p in ROOT.glob("*.md")] == ["README.md"]
    for directory in ("profiles", "configs", "baselines", "scripts", "utils", "core"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in (".py", ".json", ".sh"):
                text = path.read_text()
                assert "/home/" not in text, path
                assert "/Users/" not in text, path


@pytest.mark.parametrize("script", ("scripts/run_experiment.py", "scripts/run_baseline.py", "scripts/run_es_comparison.py", "scripts/prepare_data.py", "scripts/prepare_generator.py", "analysis/plot_results.py", "analysis/summarize_run.py"))
def test_public_entrypoints_have_help(script):
    result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
