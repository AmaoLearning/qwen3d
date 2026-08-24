#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$ROOT/.bootstrap/bin/uv"

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

if [[ ! -x "$UV" ]]; then
  retry python3 -m pip install --index-url https://pypi.org/simple --target "$ROOT/.bootstrap" uv==0.8.13
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/.python"
retry "$UV" python install 3.12
"$UV" venv --python 3.12 "$ROOT/.venv"

retry env UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$ROOT/.venv/bin/python" \
  torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu129

# Provision nvcc before building any CUDA extensions. Otherwise packages such
# as torch-scatter silently produce CPU-only shared libraries when the GPU or
# driver is not visible during environment creation.
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "$ROOT/.cuda/bin/nvcc" ]]; then
  bash "$ROOT/repro/install_cuda_toolchain.sh"
fi
source "$ROOT/repro/activate.sh"

retry env UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$ROOT/.venv/bin/python" \
  --index-url https://pypi.org/simple -r "$ROOT/requirements.txt"
retry env UV_HTTP_TIMEOUT=300 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 \
  "$UV" pip install --python "$ROOT/.venv/bin/python" \
  --index-url https://pypi.org/simple torch-scatter --no-build-isolation \
  --reinstall --no-cache
retry env UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$ROOT/.venv/bin/python" \
  --index-url https://pypi.org/simple "git+https://github.com/facebookresearch/detectron2.git@7684ebce0790bf3b8b9330b69f63c44c74a08ead" --no-build-isolation
retry env UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$ROOT/.venv/bin/python" \
  --index-url https://pypi.org/simple "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" --no-build-isolation

(cd "$ROOT" && bash docs/init.sh)

retry env UV_HTTP_TIMEOUT=300 "$UV" pip install --python "$ROOT/.venv/bin/python" \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
retry "$ROOT/.venv/bin/python" -c \
  "import nltk; nltk.download('stopwords', download_dir='$ROOT/data/nltk', raise_on_error=True)"
