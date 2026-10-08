#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ROOT_DIR}/script/.install_state"
mkdir -p "${STATE_DIR}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
# pip defaults to using a cache at ~/.cache/pip.
# - If a package is already installed, reruns usually skip it.
# - If a file finished downloading once, pip can reuse it from cache.
# - If you interrupt mid-download, that partially-downloaded file usually restarts from 0.
: "${PIP_CACHE_DIR:=${HOME}/.cache/pip}"
export PIP_CACHE_DIR

run_step() {
  local step="$1"
  shift
  local marker="${STATE_DIR}/${step}.done"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${step} (already completed)"
    return 0
  fi
  echo "[run] ${step}"
  "$@"
  touch "${marker}"
  echo "[done] ${step}"
}

run_step "requirements" bash -lc "cd \"${ROOT_DIR}\" && echo \"Installing python packages ...\" && python -m pip install -U pip && pip install -r script/requirements.txt"

run_step "cousin_matching_stack" bash -lc "set -euo pipefail; cd \"${ROOT_DIR}\" && \
echo \"Ensuring Cousin matching runtime deps ...\" && \
python -m pip install --upgrade --no-cache-dir \
  \"transformers==5.3.0\" \
  \"tokenizers==0.22.2\" \
  \"huggingface-hub==1.6.0\" \
  \"sentence-transformers==5.3.0\" && \
python - <<'PY'
import importlib.metadata as m
required = {
    'transformers': '5.3.0',
    'tokenizers': '0.22.2',
    'huggingface-hub': '1.6.0',
    'sentence-transformers': '5.3.0',
}
for pkg, want in required.items():
    got = m.version(pkg)
    if got != want:
        raise SystemExit(f\"{pkg} version mismatch: expected {want}, got {got}\")
print('[ok] Cousin matching runtime deps aligned')
PY"

run_step "system_checks" bash -lc "set -euo pipefail; cd \"${ROOT_DIR}\" && \
echo \"Checking optional system dependencies ...\" && \
missing=0; \
if ! command -v ffmpeg >/dev/null 2>&1; then \
  echo \"[warn] ffmpeg not found in PATH. Some video export / moviepy / imageio-ffmpeg workflows may fail.\"; \
  echo \"       On Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg\"; \
  missing=1; \
fi; \
if ! command -v xdotool >/dev/null 2>&1; then \
  echo \"[warn] xdotool not found in PATH. Viewer window placement (viewer_window_placement) won't work on Linux/X11.\"; \
  echo \"       On Ubuntu: sudo apt-get update && sudo apt-get install -y xdotool\"; \
  missing=1; \
fi; \
if [[ -d deps ]]; then \
  for d in GroundingDINO \"Depth-Anything-V2\" \"segment-anything-2\"; do \
    if [[ ! -d \"deps/\${d}\" ]]; then \
      echo \"[warn] Missing vendored repo: deps/\${d}. Cousin/ACDC pipelines may not run.\"; \
      missing=1; \
    fi; \
  done; \
else \
  echo \"[warn] Missing ./deps directory. Cousin/ACDC pipelines may not run.\"; \
  missing=1; \
fi; \
if [[ \"\${missing}\" -eq 0 ]]; then echo \"[ok] system checks passed\"; fi"

run_step "pytorch3d" bash -lc "set -euo pipefail; cd \"${ROOT_DIR}\" && \
python -c 'import torch; print(\"torch:\", torch.__version__)' || { \
  echo \"ERROR: torch is not importable in this environment. Re-run the requirements step (it pins torch==2.4.1) and make sure the torch wheel download finishes.\"; \
  exit 1; \
} && \
echo \"Installing pytorch3d ...\" && \
mkdir -p envs && \
PT3D_DIR=\"envs/pytorch3d\" && \
if [[ -d \"\${PT3D_DIR}/.git\" ]]; then \
  echo \"Found existing \${PT3D_DIR}; updating...\" && \
  git -C \"\${PT3D_DIR}\" fetch --all --prune && \
  (git -C \"\${PT3D_DIR}\" checkout -q stable || git -C \"\${PT3D_DIR}\" checkout -q -B stable origin/stable || true) && \
  git -C \"\${PT3D_DIR}\" pull --ff-only || true; \
else \
  rm -rf \"\${PT3D_DIR}\"; \
  CLONED=0; \
  for url in \
    \"https://github.com/facebookresearch/pytorch3d.git\" \
    \"https://gitcode.com/gh_mirror/facebookresearch/pytorch3d.git\" \
  ; do \
    echo \"Cloning pytorch3d from: \${url}\"; \
    rm -rf \"\${PT3D_DIR}\"; \
    if GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch stable --progress \"\${url}\" \"\${PT3D_DIR}\"; then \
      CLONED=1; \
      break; \
    fi; \
  done; \
  if [[ \"\${CLONED}\" -ne 1 ]]; then \
    echo \"ERROR: failed to clone pytorch3d from all sources. Try switching VPN and rerun.\"; \
    exit 1; \
  fi; \
fi && \
pip install -e \"\${PT3D_DIR}\" --no-build-isolation"

run_step "sapien_patch" bash -lc "cd \"${ROOT_DIR}\" && \
echo \"Adjusting code in sapien/wrapper/urdf_loader.py ...\" && \
SAPIEN_SITE=\$(pip show sapien >/dev/null 2>&1 && pip show sapien | awk '/^Location: /{print \$2}' || true) && \
if [[ -z \"\${SAPIEN_SITE}\" ]]; then echo \"ERROR: 'sapien' not found. Re-run after requirements install succeeds.\"; exit 1; fi && \
URDF_LOADER=\"\${SAPIEN_SITE}/sapien/wrapper/urdf_loader.py\" && \
if [[ ! -f \"\${URDF_LOADER}\" ]]; then echo \"ERROR: Missing \${URDF_LOADER}\"; exit 1; fi && \
sed -i -E 's/(\"r\")(\\))( as)/\\1, encoding=\"utf-8\") as/g' \"\${URDF_LOADER}\""

run_step "mplib_patch" bash -lc "cd \"${ROOT_DIR}\" && \
echo \"Adjusting code in mplib/planner.py ...\" && \
MPLIB_SITE=\$(pip show mplib >/dev/null 2>&1 && pip show mplib | awk '/^Location: /{print \$2}' || true) && \
if [[ -z \"\${MPLIB_SITE}\" ]]; then echo \"ERROR: 'mplib' not found. Re-run after requirements install succeeds.\"; exit 1; fi && \
PLANNER=\"\${MPLIB_SITE}/mplib/planner.py\" && \
if [[ ! -f \"\${PLANNER}\" ]]; then echo \"ERROR: Missing \${PLANNER}\"; exit 1; fi && \
sed -i -E 's/(if np.linalg.norm\\(delta_twist\\) < 1e-4 )(or collide )(or not within_joint_limit:)/\\1\\3/g' \"\${PLANNER}\""

run_step "curobo" bash -lc "set -euo pipefail; cd \"${ROOT_DIR}\" && \
echo \"Installing Curobo ...\" && \
mkdir -p envs && \
if [[ -d envs/curobo/.git ]]; then \
  echo \"Found existing envs/curobo; updating...\" && \
  git -C envs/curobo fetch --all --prune && \
  git -C envs/curobo pull --ff-only || true; \
elif [[ -d envs/curobo ]]; then \
  echo \"ERROR: envs/curobo exists but is not a git repo. Delete it or move it aside, then rerun.\"; \
  exit 1; \
else \
  CLONED=0; \
  for url in \
    \"https://github.com/NVlabs/curobo.git\" \
    \"https://gitcode.com/gh_mirror/NVlabs/curobo.git\" \
  ; do \
    echo \"Cloning curobo from: \${url}\"; \
    rm -rf envs/curobo; \
    if GIT_TERMINAL_PROMPT=0 git clone --progress \"\${url}\" envs/curobo; then \
      CLONED=1; \
      break; \
    fi; \
  done; \
  if [[ \"\${CLONED}\" -ne 1 ]]; then \
    echo \"ERROR: failed to clone curobo from all sources. Try switching VPN and rerun.\"; \
    exit 1; \
  fi; \
fi && \
cd envs/curobo && \
pip install -e . --no-build-isolation"

echo "Installation basic environment complete!"
echo "Next steps:"
echo "    1. Initialize repository submodules, if needed:"
echo "       git submodule update --init --recursive"
echo "    2. Download required simulation assets:"
echo "       bash script/_download_assets.sh"
echo "    3. See README.md for collection, cousin-layout, and optional baseline workflows."
