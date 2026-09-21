#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/repo_root.sh"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
TP="${TP:-1}"
TASKS="${TASKS:-gsm8k countdown}"
SEEDS="${SEEDS:-42 43 44}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
PYTHON="${PYTHON:-$(command -v python)}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/outputs/wall-clock}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"

GSM8K_TRAIN_DATA_PATH="${GSM8K_TRAIN_DATA_PATH:-$REPO_ROOT/data/gsm8k/train_200.parquet}"
GSM8K_TEST_DATA_PATH="${GSM8K_TEST_DATA_PATH:-$REPO_ROOT/data/gsm8k/test.parquet}"
COUNTDOWN_TRAIN_DATA_PATH="${COUNTDOWN_TRAIN_DATA_PATH:-$REPO_ROOT/data/countdown/countdown_train.json}"
COUNTDOWN_TEST_DATA_PATH="${COUNTDOWN_TEST_DATA_PATH:-$REPO_ROOT/data/countdown/countdown_validation.json}"
SENSITIVITY_PROFILE="${SENSITIVITY_PROFILE:-$REPO_ROOT/profiles/qwen2.5-1.5b-countdown-modular-shell.json}"

TRAIN_SAMPLES=200
MAX_TOKENS=1024
TOP_K=25
RANDOPT_SIGMA=0.0005
MODULAR_RADIUS=0.16
MASS_CONFIG='{"embedding":1.0,"attention":0.5,"mlp":0.5,"head":1.0,"norm":0.1,"other":0.1}'

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "GNU /usr/bin/time is required for process-level elapsed timing" >&2
  exit 1
fi
if [[ ! "$TP" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP must be a positive integer: $TP" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1: $DRY_RUN" >&2
  exit 1
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME must be 0 or 1: $RESUME" >&2
  exit 1
fi

IFS=' ' read -r -a TASK_LIST <<< "$TASKS"
IFS=' ' read -r -a SEED_LIST <<< "$SEEDS"
if (( ${#TASK_LIST[@]} == 0 || ${#SEED_LIST[@]} == 0 )); then
  echo "TASKS and SEEDS must not be empty" >&2
  exit 1
fi

for task in "${TASK_LIST[@]}"; do
  case "$task" in
    gsm8k|countdown) ;;
    *)
      echo "Unsupported task in TASKS: $task" >&2
      exit 1
      ;;
  esac
done
for seed in "${SEED_LIST[@]}"; do
  case "$seed" in
    42|43|44) ;;
    *)
      echo "Canonical wall-clock seeds are 42, 43, and 44; got $seed" >&2
      exit 1
      ;;
  esac
done

REQUIRED_FILES=("$SENSITIVITY_PROFILE")
for task in "${TASK_LIST[@]}"; do
  if [[ "$task" == "gsm8k" ]]; then
    REQUIRED_FILES+=("$GSM8K_TRAIN_DATA_PATH" "$GSM8K_TEST_DATA_PATH")
  else
    REQUIRED_FILES+=("$COUNTDOWN_TRAIN_DATA_PATH" "$COUNTDOWN_TEST_DATA_PATH")
  fi
done
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

NUM_GPUS="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
if (( NUM_GPUS % TP != 0 )); then
  echo "Number of CUDA devices ($NUM_GPUS) must be divisible by TP=$TP" >&2
  exit 1
fi
NUM_ENGINES=$((NUM_GPUS / TP))

RUN_DIR="$RESULT_ROOT/$RUN_TAG"
if [[ -e "$RUN_DIR" && "$RESUME" != "1" ]]; then
  echo "Refusing to mix measurements with an existing run directory: $RUN_DIR" >&2
  echo "Set RESUME=1 to retry incomplete runs in this RUN_TAG." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RUN_DIR/os-time" "$RUN_DIR/runs"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_TOKEN="${HF_TOKEN:-}"
export LC_ALL=C
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
unset RAY_ADDRESS || true

if [[ "$DRY_RUN" != "1" && ! -f "$RUN_DIR/hardware_before.csv" ]] \
  && command -v nvidia-smi >/dev/null 2>&1
then
  if hardware_snapshot="$(nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.limit \
    --format=csv,noheader)"
  then
    printf '%s\n' "$hardware_snapshot" > "$RUN_DIR/hardware_before.csv"
  else
    echo "Warning: nvidia-smi hardware snapshot failed" >&2
  fi
fi

MODEL_REVISION_ARGS=()
if [[ -n "$MODEL_REVISION" ]]; then
  MODEL_REVISION_ARGS=(--model_revision "$MODEL_REVISION")
fi

run_one() {
  local task="$1"
  local method="$2"
  local seed="$3"
  local train_path
  local test_path
  local population
  local radius
  local perturbation_method

  case "$task" in
    gsm8k)
      train_path="$GSM8K_TRAIN_DATA_PATH"
      test_path="$GSM8K_TEST_DATA_PATH"
      if [[ "$method" == "randopt" ]]; then
        population=300
      else
        population=25
      fi
      ;;
    countdown)
      train_path="$COUNTDOWN_TRAIN_DATA_PATH"
      test_path="$COUNTDOWN_TEST_DATA_PATH"
      if [[ "$method" == "randopt" ]]; then
        population=300
      else
        population=100
      fi
      ;;
  esac

  local -a method_args
  if [[ "$method" == "randopt" ]]; then
    radius="$RANDOPT_SIGMA"
    perturbation_method="isotropic"
    method_args=()
  else
    radius="$MODULAR_RADIUS"
    perturbation_method="recursive_modular_shell_v2"
    method_args=(
      --mass_config "$MASS_CONFIG"
      --sensitivity_profile "$SENSITIVITY_PROFILE"
      --power_iterations 8
    )
  fi

  local experiment_dir="$RUN_DIR/runs/$task/$method"
  local os_time_path="$RUN_DIR/os-time/${task}_${method}_seed${seed}.json"
  if [[ "$RESUME" == "1" ]]; then
    local -a completed_records
    shopt -s nullglob
    completed_records=(
      "$experiment_dir/${task}_${perturbation_method}_seed${seed}_"*/wall_clock.json
    )
    shopt -u nullglob
    if (( ${#completed_records[@]} > 1 )); then
      echo "Multiple completed records found for $task/$method/seed$seed" >&2
      printf '  %s\n' "${completed_records[@]}" >&2
      return 1
    fi
    if (( ${#completed_records[@]} == 1 )); then
      if [[ -f "$os_time_path" ]] \
        && grep -Eq '"exit_status"[[:space:]]*:[[:space:]]*0' "$os_time_path"
      then
        echo "Skipping completed task=$task method=$method seed=$seed"
        return
      fi
      echo "A wall_clock.json exists but successful OS timing is missing: ${completed_records[0]}" >&2
      echo "Refusing to create a duplicate completed record." >&2
      return 1
    fi
  fi
  local -a command=(
    "$PYTHON" "$REPO_ROOT/population_scaling.py"
    --dataset "$task"
    --train_data_path "$train_path"
    --test_data_path "$test_path"
    --model_name "$MODEL"
    "${MODEL_REVISION_ARGS[@]}"
    --num_engines "$NUM_ENGINES"
    --tp "$TP"
    --train_samples "$TRAIN_SAMPLES"
    --precision bfloat16
    --population_size "$population"
    --population_prefixes "$population"
    --top_k_values "$TOP_K"
    --radius "$radius"
    --perturbation_method "$perturbation_method"
    "${method_args[@]}"
    --max_tokens "$MAX_TOKENS"
    --global_seed "$seed"
    --experiment_dir "$experiment_dir"
    --cuda_devices "$CUDA_DEVICES"
    --wall_clock_mode
    --os_wall_time_path "$os_time_path"
  )

  echo "Running task=$task method=$method seed=$seed N=$population K=$TOP_K"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
    return
  fi

  /usr/bin/time \
    --quiet \
    --output="$os_time_path" \
    --format='{"elapsed_sec": %e, "user_sec": %U, "system_sec": %S, "max_rss_kb": %M, "exit_status": %x}' \
    "${command[@]}"
}

for task in "${TASK_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    # Alternate the paired method order to reduce systematic thermal/time bias.
    if [[ "$seed" == "43" ]]; then
      METHOD_ORDER=(modular_norm_randopt randopt)
    else
      METHOD_ORDER=(randopt modular_norm_randopt)
    fi
    for method in "${METHOD_ORDER[@]}"; do
      run_one "$task" "$method" "$seed"
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no measurements were executed."
  exit 0
fi

TASK_CSV="${TASKS// /,}"
SEED_CSV="${SEEDS// /,}"
"$PYTHON" "$SCRIPT_DIR/summarize_wall_clock.py" \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/summary" \
  --expected-tasks "$TASK_CSV" \
  --expected-seeds "$SEED_CSV"

if command -v nvidia-smi >/dev/null 2>&1; then
  if hardware_snapshot="$(nvidia-smi \
    --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.limit \
    --format=csv,noheader)"
  then
    printf '%s\n' "$hardware_snapshot" > "$RUN_DIR/hardware_after.csv"
  else
    echo "Warning: nvidia-smi hardware snapshot failed" >&2
  fi
fi

echo "Wall-clock experiment complete: $RUN_DIR"
echo "Summary: $RUN_DIR/summary/wall_clock_summary.md"
