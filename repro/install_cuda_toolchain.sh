#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA="$ROOT/.bootstrap/micromamba-extract/bin/micromamba"

retry() {
  local attempt=1
  until "$@"; do
    if (( attempt >= 3 )); then
      echo "command failed after 3 attempts: $*" >&2
      return 1
    fi
    echo "attempt $attempt failed; waiting 120 seconds before retry" >&2
    sleep 120
    attempt=$((attempt + 1))
  done
}

if [[ ! -x "$MAMBA" ]]; then
  mkdir -p "$ROOT/.bootstrap/micromamba-extract"
  retry curl -fL --connect-timeout 30 -o "$ROOT/.bootstrap/micromamba.tar.bz2" \
    https://micro.mamba.pm/api/micromamba/linux-64/latest
  tar -xjf "$ROOT/.bootstrap/micromamba.tar.bz2" -C "$ROOT/.bootstrap/micromamba-extract"
fi

if [[ ! -x "$ROOT/.cuda/bin/nvcc" ]]; then
  export MAMBA_ROOT_PREFIX="$ROOT/.mamba"
  retry "$MAMBA" create -y -p "$ROOT/.cuda" \
    -c nvidia/label/cuda-12.9.0 -c conda-forge \
    cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cccl=12.9 \
    libcusparse-dev=12.5.9.5 libcublas-dev=12.9.0.13 libcusolver-dev=11.7.4.40
fi

"$ROOT/.cuda/bin/nvcc" --version
