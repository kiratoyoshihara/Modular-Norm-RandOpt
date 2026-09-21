# Modular Norm RandOpt

Research code for **Modular Norm RandOpt: Population-Efficient Ensembling through Architecture-Aware Perturbations**.

Modular Norm RandOpt samples architecture-aware weight perturbations, selects experts by training reward, and ensembles their answers. It builds on [RandOpt](https://github.com/sunrainyg/RandOpt) by Yulu Gan and Phillip Isola.

## Install

Use Linux, Python 3.12, and an NVIDIA GPU. The GPU smoke tests use an RTX PRO 6000 Blackwell (96 GB); this is the tested hardware, not a minimum-memory claim. Model weights download from Hugging Face. Llama and Gemma require access to their gated repositories.

```bash
git clone https://github.com/kiratoyoshihara/Modular-Norm-RandOpt.git
cd Modular-Norm-RandOpt
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The inference environment pins PyTorch 2.10.0, vLLM 0.19.1, and Transformers 5.14.1. ES-at-Scale has a separate environment below. Choose an available GPU explicitly; the commands do not stop other users' processes.

## Data

For a Countdown quickstart, download and prepare only its public source:

```bash
python scripts/prepare_data.py --download-countdown --tasks countdown
```

For all seven tasks, obtain the [processed-data archive linked by RandOpt](https://drive.google.com/file/d/1PiAYvjZOk3VuEyGIeft7d4HynCK1lrur/view), unpack it, and point to its `data` directory:

```bash
python scripts/prepare_data.py --source /path/to/unpacked/data --tasks all
```

Preparation preserves the supplied data, derives the Countdown and MBPP splits, and verifies checksums/content identities. It never substitutes a newer dataset silently. Data/model licenses and access conditions remain those of the respective providers. Paths and completion caps are in [configs/experiments.json](configs/experiments.json); input identities are in [configs/data_manifest.json](configs/data_manifest.json). An alternative location can be passed with `--data-root`.

## Quickstart

Run two candidates on two examples to check installation. This intentionally short smoke test is **not** a paper accuracy measurement.

```bash
python scripts/run_experiment.py --suite smoke --model qwen-0.5b \
  --task countdown --method mn --gpu 0 --output outputs/smoke
```

Add `--dry-run` to inspect a command without loading data or a GPU. Every command provides `--help`. Use a fresh output directory for each run.

## Reproduce experiments

The configuration fixes model revisions, sensitivity profiles, mass allocation, perturbation strengths, prompts, completion caps, and seeds. All generation uses greedy decoding. MN/RandOpt use BF16; the iterative MeZO/ZO configurations use FP16. Candidate generation and subtractive restoration retain the RandOpt protocol.

| Experiment | Command options for `scripts/run_experiment.py` |
|---|---|
| Task/scale transfer | `--suite transfer --model qwen-0.5b` / `qwen-1.5b` / `qwen-3b` |
| Model-family transfer | `--suite transfer --model llama-3b` / `gemma-4b` / `olmo-7b`; Countdown or GSM8K |
| Population scaling | `--suite scaling --model qwen-1.5b`; Countdown or GSM8K |
| Masked ablations | `--suite ablation --model qwen-1.5b --task countdown --method attention-only` / `mlp-only` |
| RMSNorm weights only | `--suite transfer --method rmsnorm-weights-only` |
| RMSNorm scale correction | `--suite ablation --method rmsnorm-scale-correction` |
| Frobenius normalization | `--suite ablation --method frobenius` |

Tasks: `countdown`, `gsm8k`, `mbpp`, `rocstories`, `uspto50k`, `math500`, `olympiadbench`. Methods `mn` and `randopt` share the same population/ensemble settings. Repeat each paper configuration with seeds 42, 43, and 44:

```bash
for seed in 42 43 44; do
  python scripts/run_experiment.py --suite transfer --model qwen-1.5b \
    --task gsm8k --method mn --seed "$seed" --gpu 0 \
    --output "outputs/gsm8k-mn-$seed"
done
```

MBPP executes generated Python; run it in an isolated environment without sensitive files.

`attention-only` and `mlp-only` retain full-method scales and mask other perturbations without renormalization. `rmsnorm-weights-only` perturbs only RMSNorm weights; `rmsnorm-scale-correction` instead changes the scale construction while perturbing the full parameter set.

Each run saves arguments, candidate rewards, selected candidates, predictions, timing, and `paper_metrics.json`. Paper metrics use exact ordering accuracy for ROCStories, balanced accuracy for USPTO-50K, and the task handlers for other tasks. Selection rewards are unchanged. Lower-level options, including arbitrary population prefixes and runtime accounting, are available through `python population_scaling.py --help`.

### Calibration

Published profiles are included. To regenerate a profile and select a cross-family radius, use the corresponding pinned model/revision from the experiment configuration:

```bash
python scripts/calibrate_decoder_sensitivities.py --model_name MODEL \
  --model_revision REVISION --data_path data/countdown/countdown_train.json \
  --output outputs/profile.json
python scripts/measure_functional_displacement.py --model_name MODEL \
  --model_revision REVISION --train_data_path data/countdown/countdown_train.json \
  --sensitivity_profile outputs/profile.json --output outputs/radius.json
```

These commands use the shared calibration implementation; `--help` lists the radius grid, prompt count, and power-iteration settings.
For Qwen profile regeneration use `scripts/calibrate_qwen_sensitivities.py`. For OLMo radius selection use `scripts/functional_displacement/measure_olmo3_functional_displacement.py`, which handles its padded vocabulary.

Runtime comparisons: `CUDA_DEVICES=0 bash scripts/wall_clock/run_wall_clock_k25.sh` runs the paired K=25 configurations and includes process-level timing. `DRY_RUN=1` prints the commands without GPU work.

### MeZO and ZO-Finetuner

The standalone runner shares the original update kernels and uses vLLM for generation. [configs/iterative.json](configs/iterative.json) contains the selected configuration and tuning grids. The small trained ZO perturbation generator is included in `artifacts/`, with a checksum and preparation settings.

```bash
python scripts/run_baseline.py --method zo-finetuner --task countdown \
  --phase final --seed 42 --gpu 0 --output outputs/zo-countdown-42
python scripts/run_baseline.py --method mezo --task gsm8k \
  --phase final --seed 42 --gpu 0 --output outputs/mezo-gsm8k-42
```

Repeat for both tasks and seeds 42–44. Each final run performs 1,500 two-sided updates on 200 training prompts, evaluates development data every 100 updates including step zero, then tests its best development checkpoint once. Ties prefer higher reward, then the earlier checkpoint. Results record training, development, and test evaluation counts separately.

For tuning, use `--phase calibration --task countdown --seed 39 --learning-rate VALUE` for each method's grid; each run uses 500 updates. Among runs marked `stable` (terminal development accuracy drops at most 5 pp from step zero), rank their best development scores by accuracy, then reward, then smaller learning rate. `--phase smoke` runs one update on two examples. No prior experiment queue is required.

Generator preparation can be reproduced separately; it uses supervised Countdown solutions and is not part of the downstream 600,000 training-evaluation budget:

```bash
python scripts/prepare_generator.py --gpu 0 --output outputs/generator
# Use its output with: --generator outputs/generator/meta/generator.pt
```

Preparation uses FP32 and saves full training checkpoints; allow at least 25 GiB of free disk space in addition to model downloads. It is unnecessary when using the included generator.

### Iterative ES

ES-at-Scale uses its pinned upstream implementation and a separate environment:

```bash
bash scripts/baselines/es_at_scale/bootstrap_env.sh
venvs/es-at-scale/bin/python scripts/baselines/es_at_scale/run_es_baseline.py \
  --task countdown --phase final --seed 42 --cuda-devices 0 \
  --mini-batch-size 200 --selection-artifact configs/es_selection.json \
  --output-root outputs/es
```

Repeat for GSM8K and seeds 42–44. This uses 30 perturbations × 100 iterations. For recalibration, run `--phase calibration --task countdown --sigma VALUE --seed SEED` for σ ∈ {0.0005, 0.001, 0.002} and seeds 39–41, then select with `analysis/es/summarize_runs.py --input-root outputs/es --output-dir outputs/es-selection --select`.

The corresponding MN comparison uses the main inference environment. First run `scripts/run_es_comparison.py --phase search --task countdown --seed 42 --gpu 0 --output outputs/mn-search`; then run it with `--phase ensemble --source-run outputs/mn-search/runs/RUN_DIRECTORY --output outputs/mn-ensemble` and the same task/seed. The second step consumes that new search's saved candidates, not any private log directory. Repeat for GSM8K and seeds 42–44.

## Figures and tests (CPU)

Published per-seed values, candidate rewards, and figure inputs are in `results/`. Numerical plots can be regenerated without GPU access, model downloads, or private experiment logs. This compact renderer preserves the plotted values and mean ± sample-SD convention; typography/layout may differ from the paper artwork.

```bash
python analysis/plot_results.py --output outputs/figures
python -m pytest -q
```

CPU-only installation: install `torch==2.10.0` from the [PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu), then `pip install -r requirements-test.txt`. No GPU experiment starts during the tests. GitHub Actions runs the same CPU suite and figure generation.

## Credits and citation

This implementation builds on [RandOpt / Neural Thickets](https://github.com/sunrainyg/RandOpt) (Yulu Gan and Phillip Isola), [ES-at-Scale](https://github.com/VsonicV/es-at-scale), [MeZO](https://github.com/princeton-nlp/MeZO), and [ZO-Finetuner](https://github.com/ASTRAL-Group/ZO_Fine_tuner). ZO-Finetuner here is a task-adapted implementation, including supervised generator preparation, not a reproduction of the original paper's task suite. Upstream authorship and applicable third-party terms are preserved; model and dataset terms are separate.

Third-party notices: [MeZO (MIT)](third_party/MeZO-LICENSE.txt) and [ES-at-Scale (Academic Public License)](third_party/ES-at-Scale-LICENSE.txt). The latter has separate commercial-use terms; these are not a blanket license for this repository.

```bibtex
@misc{gan2026neuralthickets,
  title={Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights},
  author={Yulu Gan and Phillip Isola},
  year={2026},
  eprint={2603.12228},
  archivePrefix={arXiv}
}
```
