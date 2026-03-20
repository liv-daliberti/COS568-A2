#!/bin/bash
#SBATCH --job-name=cos568-task2a-rte-cpu
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR_OVERRIDE:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_DIR" ]]; then
  echo "REPO_DIR_OVERRIDE or SLURM_SUBMIT_DIR must be set" >&2
  exit 1
fi
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/outputs"

if [[ -z "${VENV_PATH:-}" && -d "$REPO_DIR/.venv_cos568" ]]; then
  VENV_PATH="$REPO_DIR/.venv_cos568"
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

python3 -c "import torch, pytorch_transformers, numpy, sklearn, tqdm"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_DIR/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TMPDIR="${TMPDIR:-$REPO_DIR/.tmp}"

export GLUE_DIR="${GLUE_DIR:-$REPO_DIR/glue_data}"
export TASK_NAME="${TASK_NAME:-RTE}"
export SYNC_METHOD="${SYNC_METHOD:-gather_scatter}"
export ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
export PYTORCH_TRANSFORMERS_CACHE="${PYTORCH_TRANSFORMERS_CACHE:-$REPO_DIR/hf_cache}"
if [[ "$SYNC_METHOD" == "all_reduce" ]]; then
  DEFAULT_OUTPUT_TAG="task2b_cpu"
elif [[ "$SYNC_METHOD" == "ddp" ]]; then
  DEFAULT_OUTPUT_TAG="task3_cpu"
else
  DEFAULT_OUTPUT_TAG="task2a_cpu"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/${TASK_NAME}_${DEFAULT_OUTPUT_TAG}_ws${SLURM_NTASKS}}"
export PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$OUTPUT_DIR/profiler}"
export PROFILE_WAIT_STEPS="${PROFILE_WAIT_STEPS:-1}"
export PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-3}"
export RUN_GLUE_EXTRA_ARGS="${RUN_GLUE_EXTRA_ARGS:-}"
export CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"

if [[ "$CLEAN_OUTPUT_DIR" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
  rm -rf "$PROFILE_OUTPUT_DIR"
fi

mkdir -p "$GLUE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PYTORCH_TRANSFORMERS_CACHE" "$OUTPUT_DIR" "$TMPDIR"
if [[ ! -d "$GLUE_DIR/$TASK_NAME" ]]; then
  python3 download_glue_data.py --data_dir "$GLUE_DIR" --tasks "$TASK_NAME"
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
ALLOCATED_NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
MASTER_PORT="${MASTER_PORT:-29511}"
WORLD_SIZE="${WORLD_SIZE:-$SLURM_NTASKS}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"  # 16 * 4 = 64 total batch size

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE (total batch size = $((PER_DEVICE_TRAIN_BATCH_SIZE * WORLD_SIZE)))"
echo "SYNC_METHOD=$SYNC_METHOD"
echo "ENABLE_PROFILING=$ENABLE_PROFILING PROFILE_OUTPUT_DIR=$PROFILE_OUTPUT_DIR"

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
    "world_size": int("$WORLD_SIZE"),
    "master_addr": "$MASTER_ADDR",
    "master_port": "$MASTER_PORT",
    "sync_method": "$SYNC_METHOD",
    "enable_profiling": "$ENABLE_PROFILING" == "1",
    "uses_cuda": False,
    "output_dir": "$OUTPUT_DIR",
    "profile_output_dir": "$PROFILE_OUTPUT_DIR",
}
Path("$OUTPUT_DIR/run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

srun --ntasks="$WORLD_SIZE" --ntasks-per-node=1 --label bash -lc '
set -euo pipefail
cd "'"$REPO_DIR"'"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi
export XDG_CACHE_HOME="'"$XDG_CACHE_HOME"'"
export HF_HOME="'"$HF_HOME"'"
export TRANSFORMERS_CACHE="'"$TRANSFORMERS_CACHE"'"
export TORCH_HOME="'"$TORCH_HOME"'"
export PYTORCH_TRANSFORMERS_CACHE="'"$PYTORCH_TRANSFORMERS_CACHE"'"
export TMPDIR="'"$TMPDIR"'"
echo "RANK=$SLURM_PROCID HOST=$(hostname)"

PROFILE_ARGS=()
if [[ "${ENABLE_PROFILING}" == "1" ]]; then
  PROFILE_ARGS+=(--enable_profiling)
  PROFILE_ARGS+=(--profile_output_dir "${PROFILE_OUTPUT_DIR}")
  PROFILE_ARGS+=(--profile_wait_steps "${PROFILE_WAIT_STEPS}")
  PROFILE_ARGS+=(--profile_active_steps "${PROFILE_ACTIVE_STEPS}")
fi

EXTRA_ARGS=()
if [[ -n "${RUN_GLUE_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${RUN_GLUE_EXTRA_ARGS}"
fi

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name "'"$TASK_NAME"'" \
  --do_train \
  --do_eval \
  --data_dir "'"$GLUE_DIR"'"/"'"$TASK_NAME"'" \
  --max_seq_length 128 \
  --per_device_train_batch_size "'"$PER_DEVICE_TRAIN_BATCH_SIZE"'" \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir "'"$OUTPUT_DIR"'" \
  --cache_dir "'"$PYTORCH_TRANSFORMERS_CACHE"'" \
  --overwrite_output_dir \
  --no_cuda \
  --master_ip "'"$MASTER_ADDR"'" \
  --master_port "'"$MASTER_PORT"'" \
  --world_size "'"$WORLD_SIZE"'" \
  --local_rank "$SLURM_PROCID" \
  "${PROFILE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --sync_method "'"$SYNC_METHOD"'"
'
