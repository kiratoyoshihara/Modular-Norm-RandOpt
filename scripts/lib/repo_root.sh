#!/usr/bin/env bash

# Resolve the repository root from this helper's own location so launchers can
# be invoked from any working directory after being organized into subfolders.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "$REPO_ROOT/population_scaling.py" || ! -d "$REPO_ROOT/utils" ]]; then
  echo "Could not resolve Modular-Norm-RandOpt repository root: $REPO_ROOT" >&2
  return 1
fi

cd "$REPO_ROOT"
