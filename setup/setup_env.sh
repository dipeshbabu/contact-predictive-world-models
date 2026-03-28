#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JAX_TARGET="${1:-cpu}"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required, found {sys.version.split()[0]}")
PY

# Cloud images often come with a partially incompatible JAX/TensorFlow stack
# preinstalled. Remove the overlapping packages first so the pinned project
# environment is resolved from a clean slate.
python -m pip uninstall -y \
  chex \
  dm-sonnet \
  jax \
  jaxlib \
  jax-cuda12-pjrt \
  jax-cuda12-plugin \
  ml-dtypes \
  optax \
  orbax-checkpoint \
  orbax-export \
  tensorflow \
  tensorflow-probability || true

python -m pip install --upgrade pip setuptools wheel

case "$JAX_TARGET" in
  cpu)
    python -m pip install "jax[cpu]==0.4.28"
    ;;
  cuda12|gpu)
    python -m pip install "jax[cuda12]==0.4.28"
    ;;
  *)
    echo "usage: bash setup/setup_env.sh [cpu|cuda12|gpu]" >&2
    exit 1
    ;;
esac

python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
