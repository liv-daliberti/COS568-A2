#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file instead:"
  echo "  source ./env_local.sh"
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export PYTORCH_TRANSFORMERS_CACHE="${PYTORCH_TRANSFORMERS_CACHE:-$REPO_DIR/hf_cache}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.tmp}"
export GLUE_DIR="${GLUE_DIR:-$REPO_DIR/glue_data}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PYTORCH_TRANSFORMERS_CACHE" "$TMPDIR" "$GLUE_DIR" "$REPO_DIR/logs" "$REPO_DIR/outputs"

if [[ -d "$REPO_DIR/.venv_cos568" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_DIR/.venv_cos568/bin/activate"
fi
