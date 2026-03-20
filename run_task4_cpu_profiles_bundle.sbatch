#!/bin/bash
#SBATCH --job-name=cos568-task4-cpu-bundle
#SBATCH --partition=lowprio
#SBATCH --account=mltheory
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/outputs/task4_cpu_nodes}"
mkdir -p "$OUTPUT_ROOT"

run_profile() {
  local sync_method="$1"
  local master_port="$2"
  local output_dir="$3"
  local profile_output_dir="$4"

  echo "=== CPU profile run: method=$sync_method port=$master_port output_dir=$output_dir ==="
  SYNC_METHOD="$sync_method" \
  ENABLE_PROFILING=1 \
  RUN_GLUE_EXTRA_ARGS="--max_steps 4" \
  OUTPUT_DIR="$output_dir" \
  PROFILE_OUTPUT_DIR="$profile_output_dir" \
  MASTER_PORT="$master_port" \
  bash "$REPO_DIR/run_task2a_4workers_cpu.sbatch"
}

run_profile "gather_scatter" "29831" "$OUTPUT_ROOT/gather_scatter/run" "$OUTPUT_ROOT/gather_scatter/traces"
run_profile "all_reduce" "29832" "$OUTPUT_ROOT/all_reduce/run" "$OUTPUT_ROOT/all_reduce/traces"
run_profile "ddp" "29833" "$OUTPUT_ROOT/ddp/run" "$OUTPUT_ROOT/ddp/traces"
