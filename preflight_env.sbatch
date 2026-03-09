#!/bin/bash
#SBATCH --job-name=cos568-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

if [[ -z "${VENV_PATH:-}" && -d "$REPO_DIR/.venv_cos568" ]]; then
  VENV_PATH="$REPO_DIR/.venv_cos568"
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PYTORCH_TRANSFORMERS_CACHE="${PYTORCH_TRANSFORMERS_CACHE:-$REPO_DIR/hf_cache}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.tmp}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PYTORCH_TRANSFORMERS_CACHE" "$TMPDIR"

echo "HOST=$(hostname)"
echo "PATH=$PATH"
echo "REPO_DIR=$REPO_DIR"
echo "VENV_PATH=${VENV_PATH:-<none>}"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "TORCH_HOME=$TORCH_HOME"
echo "PYTORCH_TRANSFORMERS_CACHE=$PYTORCH_TRANSFORMERS_CACHE"
echo "TMPDIR=$TMPDIR"
which python3 || true
python3 -V || true
python3 -c "import sys; print('sys.executable=', sys.executable)" || true

python3 - <<'PY'
mods = ["torch", "pytorch_transformers", "numpy", "sklearn", "tqdm"]
for m in mods:
    try:
        __import__(m)
        print(f"IMPORT_OK {m}")
    except Exception as e:
        print(f"IMPORT_FAIL {m}: {e}")
PY

nvidia-smi || true
