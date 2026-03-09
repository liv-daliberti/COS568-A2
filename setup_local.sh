#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv_cos568}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PYTORCH_TRANSFORMERS_CACHE="${PYTORCH_TRANSFORMERS_CACHE:-$REPO_DIR/hf_cache}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.tmp}"

mkdir -p logs outputs glue_data "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PYTORCH_TRANSFORMERS_CACHE" "$TMPDIR"

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "${UPGRADE_PIP:-0}" == "1" ]]; then
  python3 -m pip install --upgrade pip
fi

MISSING_MODULES="$(
python3 - <<'PY'
import importlib.util

modules = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "sklearn",
    "tqdm",
    "pytorch_transformers",
]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"

if [[ -n "$MISSING_MODULES" ]]; then
  echo "Installing missing python modules: $MISSING_MODULES"
  python3 -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install numpy scipy scikit-learn tqdm pytorch_transformers
fi

if [[ ! -d "$REPO_DIR/glue_data/RTE" ]]; then
  SHARED_GLUE_DIR="/n/fs/similarity/kalshi/glue_data"
  if [[ -d "$SHARED_GLUE_DIR/RTE" ]]; then
    cp -a "$SHARED_GLUE_DIR/RTE" "$REPO_DIR/glue_data/"
  else
    python3 download_glue_data.py --data_dir "$REPO_DIR/glue_data" --tasks RTE
  fi
fi

if ! compgen -G "$PYTORCH_TRANSFORMERS_CACHE/*" >/dev/null; then
  SHARED_HF_CACHE="/n/fs/similarity/kalshi/hf_cache"
  if [[ -d "$SHARED_HF_CACHE" ]] && compgen -G "$SHARED_HF_CACHE/*" >/dev/null; then
    cp -a "$SHARED_HF_CACHE"/. "$PYTORCH_TRANSFORMERS_CACHE"/
  fi
fi

python3 - <<'PY'
import os
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer

cache_dir = os.environ["PYTORCH_TRANSFORMERS_CACHE"]
BertConfig.from_pretrained("bert-base-cased", cache_dir=cache_dir)
BertTokenizer.from_pretrained("bert-base-cased", cache_dir=cache_dir)
BertForSequenceClassification.from_pretrained("bert-base-cased", cache_dir=cache_dir)

print(f"Local setup complete. Model cache: {cache_dir}")
PY

cat <<EOF
All local paths are now under:
  $REPO_DIR

Next:
  source ./env_local.sh
  python3 run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name RTE --do_train --do_eval --data_dir "$REPO_DIR/glue_data/RTE" --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 3 --output_dir "$REPO_DIR/outputs/RTE" --cache_dir "$PYTORCH_TRANSFORMERS_CACHE" --overwrite_output_dir
EOF
