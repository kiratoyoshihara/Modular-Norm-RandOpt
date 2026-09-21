# Modular Norm RandOpt

**Population-efficient ensembling through architecture-aware perturbations.**

[Project page](https://kiratoyoshihara.github.io/Modular-Norm-RandOpt-page/) · [Experiment settings](configs/experiments.json) · [Results](results/)

[![Modular Norm RandOpt: sample noise, apply module-wise scaling, select top-K candidates, and vote.](assets/method.gif)](https://kiratoyoshihara.github.io/Modular-Norm-RandOpt-page/#overview)

## Quickstart

Linux, Python 3.12, and an NVIDIA GPU. Dependencies are pinned in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/kiratoyoshihara/Modular-Norm-RandOpt.git
cd Modular-Norm-RandOpt
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py --download-countdown --tasks countdown
python scripts/run_experiment.py --suite smoke --model qwen-0.5b \
  --task countdown --method mn --gpu 0 --output outputs/smoke
```

This checks installation with two candidates on two examples, not paper accuracy. Choose an available GPU and a fresh output directory; add `--dry-run` to preview commands.

## Experiments

Download the [processed-data archive](https://drive.google.com/file/d/1PiAYvjZOk3VuEyGIeft7d4HynCK1lrur/view) for all seven tasks, then run:

```bash
python scripts/prepare_data.py --source /path/to/unpacked/data --tasks all
python scripts/run_experiment.py --suite transfer --model qwen-1.5b \
  --task gsm8k --method mn --seed 42 --gpu 0 --output outputs/gsm8k-mn-42
```

Use `--method randopt` for the baseline; repeat with seeds 42–44. Models, tasks, and fixed settings are in [configs/experiments.json](configs/experiments.json). Runs save predictions and `paper_metrics.json`.

| Experiment | Entry point |
|---|---|
| Population scaling / ablations | [run_experiment.py](scripts/run_experiment.py): `--suite scaling` / `--suite ablation` |
| MeZO / task-adapted ZO-Finetuner | [run_baseline.py](scripts/run_baseline.py) · [settings](configs/iterative.json) |
| Iterative ES | [environment setup](scripts/baselines/es_at_scale/bootstrap_env.sh) · [runner](scripts/baselines/es_at_scale/run_es_baseline.py) |
| Runtime comparison | [run_wall_clock_k25.sh](scripts/wall_clock/run_wall_clock_k25.sh) |

Python entry points provide `--help`. [Sensitivity profiles](profiles/) and the [ZO generator](artifacts/) are included. Llama/Gemma require model access; MBPP executes generated Python and should run in an isolated environment.

## Figures and tests

Replot Figures 1–5 from the included CSVs; no private logs or GPU are needed. Layout may differ from the paper.

```bash
python analysis/plot_results.py --output outputs/figures
python -m pytest -q
```

For CPU-only installation, install [PyTorch's CPU wheel](https://download.pytorch.org/whl/cpu) first, then `pip install -r requirements-test.txt`.

## Acknowledgments

Built on [RandOpt](https://github.com/sunrainyg/RandOpt) by Yulu Gan and Phillip Isola, with [ES-at-Scale](https://github.com/VsonicV/es-at-scale), [MeZO](https://github.com/princeton-nlp/MeZO), and task-adapted [ZO-Finetuner](https://github.com/ASTRAL-Group/ZO_Fine_tuner) baselines.

See [third-party licenses](third_party/); ES-at-Scale has separate commercial-use terms. Model and dataset terms apply separately.
