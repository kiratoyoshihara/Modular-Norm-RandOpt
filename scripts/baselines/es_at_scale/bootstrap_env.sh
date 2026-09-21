#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.12}"
VENV_DIR="${VENV_DIR:-venvs/es-at-scale}"
UPSTREAM_DIR="${ES_AT_SCALE_SOURCE:-external/es-at-scale}"
UPSTREAM_REPOSITORY="https://github.com/VsonicV/es-at-scale.git"
UPSTREAM_COMMIT="574a9d134da1ffce2a8bb812019899e5c96b588a"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-es.lock

if [[ ! -e "$UPSTREAM_DIR" ]]; then
  git clone "$UPSTREAM_REPOSITORY" "$UPSTREAM_DIR"
  git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_COMMIT"
fi

actual_commit="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$UPSTREAM_COMMIT" ]]; then
  echo "Unexpected ES-at-Scale commit in $UPSTREAM_DIR" >&2
  echo "expected=$UPSTREAM_COMMIT" >&2
  echo "actual=$actual_commit" >&2
  exit 1
fi

if [[ -n "$(git -C "$UPSTREAM_DIR" status --porcelain)" ]]; then
  echo "ES-at-Scale source has local modifications: $UPSTREAM_DIR" >&2
  exit 1
fi

ES_AT_SCALE_SOURCE="$UPSTREAM_DIR" \
  "$VENV_DIR/bin/python" scripts/baselines/es_at_scale/run_es_baseline.py --help >/dev/null

echo "ES-at-Scale environment is ready."
echo "python=$VENV_DIR/bin/python"
echo "source=$UPSTREAM_DIR"
echo "commit=$actual_commit"
