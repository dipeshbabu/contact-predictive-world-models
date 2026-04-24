#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JAX_TARGET="${1:-cpu}"
INSTALLER="${CPWM_INSTALLER:-uv}"

PYTHON_EXE="$(
  python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Python 3.11 is required, found {sys.version.split()[0]}")
print(sys.executable)
PY
)"

echo "Using project Python: ${PYTHON_EXE}"

case "$INSTALLER" in
  uv)
    if ! command -v uv >/dev/null 2>&1; then
      echo "uv is required by default. Install uv, or run with CPWM_INSTALLER=pip." >&2
      exit 1
    fi
    PIP_INSTALL=(uv pip install --python "${PYTHON_EXE}")
    PIP_UNINSTALL=(uv pip uninstall --python "${PYTHON_EXE}")
    PIP_CHECK=(uv pip check --python "${PYTHON_EXE}")
    ;;
  pip)
    PIP_INSTALL=("${PYTHON_EXE}" -m pip install)
    PIP_UNINSTALL=("${PYTHON_EXE}" -m pip uninstall -y)
    PIP_CHECK=("${PYTHON_EXE}" -m pip check)
    ;;
  *)
    echo "CPWM_INSTALLER must be 'uv' or 'pip', found '$INSTALLER'" >&2
    exit 1
    ;;
esac

if ! "${PYTHON_EXE}" -m pip --version >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_EXE}" pip setuptools wheel
  else
    "${PYTHON_EXE}" -m ensurepip --upgrade
  fi
fi

# Cloud images often come with a partially incompatible JAX/TensorFlow stack
# preinstalled. Remove the overlapping packages first so the pinned project
# environment is resolved from a clean slate.
"${PIP_UNINSTALL[@]}" \
  brax \
  chex \
  contact-predictive-world-models \
  dm-sonnet \
  distrax \
  flax \
  gymnax \
  humanoid-bench \
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

"${PIP_INSTALL[@]}" --upgrade pip setuptools wheel

case "$JAX_TARGET" in
  cpu)
    "${PIP_INSTALL[@]}" "jax[cpu]==0.4.28"
    ;;
  cuda12|gpu)
    "${PIP_INSTALL[@]}" "jax[cuda12]==0.4.28"
    ;;
  *)
    echo "usage: bash setup/setup_env.sh [cpu|cuda12|gpu]" >&2
    exit 1
    ;;
esac

"${PIP_INSTALL[@]}" -r requirements.txt
"${PIP_INSTALL[@]}" -e . --no-deps
"${PIP_CHECK[@]}"
