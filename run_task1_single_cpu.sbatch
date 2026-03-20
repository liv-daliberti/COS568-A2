#!/bin/bash
#SBATCH --job-name=cos568-task1-rte-cpu
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_DIR" ]]; then
  echo "REPO_DIR_OVERRIDE or SLURM_SUBMIT_DIR must be set" >&2
  exit 1
fi
cd "$REPO_DIR"

mkdir -p "$REPO_DIR/logs"

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
export TMPDIR="${TMPDIR:-$REPO_DIR/.tmp}"

export GLUE_DIR="${GLUE_DIR:-$REPO_DIR/glue_data}"
export TASK_NAME="${TASK_NAME:-RTE}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/${TASK_NAME}_task1_cpu}"
export PYTORCH_TRANSFORMERS_CACHE="${PYTORCH_TRANSFORMERS_CACHE:-$REPO_DIR/hf_cache}"
export CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"

python3 -c "import torch, pytorch_transformers, numpy, sklearn, tqdm"

if [[ ! -d "$GLUE_DIR/$TASK_NAME" ]]; then
  mkdir -p "$GLUE_DIR"
  python3 download_glue_data.py --data_dir "$GLUE_DIR" --tasks "$TASK_NAME"
fi

if [[ "$CLEAN_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PYTORCH_TRANSFORMERS_CACHE" "$TMPDIR"

ALLOCATED_NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
python3 - <<PY
import json
from pathlib import Path

nodes = """$ALLOCATED_NODES""".split()
metadata = {
    "job_id": "$SLURM_JOB_ID",
    "job_name": "$SLURM_JOB_NAME",
    "partition": "$SLURM_JOB_PARTITION",
    "account": "$SLURM_JOB_ACCOUNT",
    "nodes": nodes,
    "num_nodes": len(nodes),
    "ntasks": int("${SLURM_NTASKS:-1}"),
    "ntasks_per_node": int("${SLURM_NTASKS_PER_NODE:-1}"),
    "cpus_per_task": int("${SLURM_CPUS_PER_TASK:-1}"),
    "uses_cuda": False,
    "sync_method": "none",
    "output_dir": "$OUTPUT_DIR",
}
Path("$OUTPUT_DIR/run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name "$TASK_NAME" \
  --do_train \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$PYTORCH_TRANSFORMERS_CACHE" \
  --overwrite_output_dir \
  --no_cuda
