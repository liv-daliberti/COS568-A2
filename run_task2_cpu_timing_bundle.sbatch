#!/bin/bash
#SBATCH --job-name=cos568-task2-cpu-bundle
#SBATCH --partition=cs
#SBATCH --account=allcs
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

export REPO_DIR_OVERRIDE="$REPO_DIR"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/report_runs_cpu_nodes}"
mkdir -p "$OUTPUT_ROOT"

run_timing() {
  local sync_method="$1"
  local master_port="$2"
  local output_dir="$3"

  echo "=== CPU timing run: method=$sync_method port=$master_port output_dir=$output_dir ==="
  SYNC_METHOD="$sync_method" \
  ENABLE_PROFILING=0 \
  OUTPUT_DIR="$output_dir" \
  MASTER_PORT="$master_port" \
  bash "$REPO_DIR/run_task2a_4workers_cpu.sbatch"
}

run_timing "gather_scatter" "29611" "$OUTPUT_ROOT/task2a"
run_timing "all_reduce" "29612" "$OUTPUT_ROOT/task2b"
run_timing "ddp" "29613" "$OUTPUT_ROOT/task3"
