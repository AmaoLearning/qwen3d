#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/repro/activate.sh"
VENV_PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1

retry() {
  local attempt=1
  until "$@"; do
    if ((attempt >= 3)); then
      echo "command failed after 3 attempts: $*" >&2
      return 1
    fi
    echo "attempt $attempt failed; retrying in 120s: $*" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
}

if "$VENV_PY" - <<'PY'
import importlib.util
import sys
raise SystemExit(0 if importlib.util.find_spec("pointops2") is not None else 1)
PY
then
  echo "pointops2 is already available"
  exit 0
fi

if command -v nvcc >/dev/null 2>&1; then
  :
elif [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  :
else
  echo "CUDA compiler not found (nvcc). If you don't have CUDA installed, install it first or run install_cuda_toolchain.sh."
  echo "Current CUDA_HOME=${CUDA_HOME:-<unset>}"
  exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "Python 3.12 virtualenv not found at $VENV_PY"
  exit 1
fi

if [[ "$("$VENV_PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" != "3.12" ]]; then
  echo "Virtualenv Python is not 3.12, refuse to continue."
  "$VENV_PY" -c 'import sys; print(sys.executable, sys.version)'
  exit 1
fi

cd "$ROOT/libs/pointops2"
rm -rf build
retry env PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT" "$VENV_PY" setup.py install
cd "$ROOT"

if ! "$VENV_PY" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pointops2") is not None else 1)
PY
then
  echo "pointops2 still not importable after build"
  exit 1
fi

echo "pointops2 installed successfully"
