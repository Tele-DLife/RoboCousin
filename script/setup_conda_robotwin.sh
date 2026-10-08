#!/usr/bin/env bash
# One-shot: create (or reuse) conda env "RoboTwin" and run script/_install.sh
#
# Environment variables (optional):
#   CONDA_ENV_NAME   default: RoboTwin
#   PYTHON_VERSION   default: 3.10
#
# Flags:
#   --recreate       remove existing env and create again (also clears script/.install_state)
#   --env-only       only create/update the conda env + pip base tools; skip _install.sh
#
# After success:
#   conda activate RoboTwin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-RoboTwin}"
PY_VER="${PYTHON_VERSION:-3.10}"
RECREATE=0
ENV_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    --env-only) ENV_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found. Install Miniconda or Anaconda and ensure 'conda' is on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

env_exists() {
  conda env list | awk '!/^#/ && NF {print $1}' | grep -qx "$1"
}

if [[ "$RECREATE" -eq 1 ]]; then
  if env_exists "$ENV_NAME"; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove -n "$ENV_NAME" --all -y
  fi
  rm -rf "${ROOT}/script/.install_state"
fi

if env_exists "$ENV_NAME"; then
  echo "Using existing environment: $ENV_NAME"
  conda activate "$ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME (python=${PY_VER})"
  conda create -n "$ENV_NAME" "python=${PY_VER}" -y
  conda activate "$ENV_NAME"
fi

python -m pip install -U pip setuptools wheel

if [[ "$ENV_ONLY" -eq 1 ]]; then
  echo "Skipping script/_install.sh (--env-only)."
  echo "Activate later with: conda activate ${ENV_NAME}"
  exit 0
fi

bash "${ROOT}/script/_install.sh"

echo ""
echo "RoboTwin conda setup finished."
echo "Activate with: conda activate ${ENV_NAME}"
