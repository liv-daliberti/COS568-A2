#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclass
class JobSpec:
    name: str
    script: Path
    env: dict[str, str]


def submit_job(spec: JobSpec, dependency: str | None = None, extra_sbatch_args: list[str] | None = None) -> str:
    env = os.environ.copy()
    env["REPO_DIR_OVERRIDE"] = str(REPO_DIR)
    env.update(spec.env)
    command = ["sbatch", "--parsable"]
    if extra_sbatch_args:
        command.extend(extra_sbatch_args)
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(spec.script))
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    job_id = result.stdout.strip().split(";", 1)[0]
    print(f"submitted {spec.name}: job {job_id}")
    return job_id


def query_job_states(job_ids: list[str]) -> dict[str, str]:
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            ",".join(job_ids),
            "-o",
            "JobIDRaw,State",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id, state = parts[0], parts[1]
        base_state = state.split("+", 1)[0]
        if job_id in job_ids:
            states[job_id] = base_state
    return states


def wait_for_jobs(job_ids: list[str], poll_seconds: int) -> dict[str, str]:
    pending = set(job_ids)
    last_states: dict[str, str] = {}
    while pending:
        states = query_job_states(job_ids)
        for job_id in list(pending):
            state = states.get(job_id)
            if not state:
                continue
            last_states[job_id] = state
            if state in TERMINAL_STATES:
                pending.remove(job_id)
        status_line = ", ".join(f"{job_id}={last_states.get(job_id, 'PENDING')}" for job_id in job_ids)
        print(status_line)
        if pending:
            time.sleep(poll_seconds)
    return last_states


def build_job_specs(include_task1: bool) -> tuple[list[JobSpec], list[JobSpec]]:
    cpu_root = REPO_DIR / "outputs" / "report_runs_cpu_nodes"
    gpu_root = REPO_DIR / "outputs" / "report_runs_gpu_nodes"
    cpu_profile_root = REPO_DIR / "outputs" / "task4_cpu_nodes"
    gpu_profile_root = REPO_DIR / "outputs" / "task4_gpu_nodes"

    cpu_specs = []
    if include_task1:
        cpu_specs.append(
            JobSpec(
                name="cpu_task1",
                script=REPO_DIR / "run_task1_single_cpu.sbatch",
                env={"OUTPUT_DIR": str(cpu_root / "task1")},
            )
        )
    cpu_specs.extend([
        JobSpec(
            name="cpu_task2_bundle",
            script=REPO_DIR / "run_task2_cpu_timing_bundle.sbatch",
            env={"OUTPUT_ROOT": str(cpu_root)},
        ),
        JobSpec(
            name="cpu_task4_bundle",
            script=REPO_DIR / "run_task4_cpu_profiles_bundle.sbatch",
            env={"OUTPUT_ROOT": str(cpu_profile_root)},
        ),
    ])

    gpu_specs = []
    if include_task1:
        gpu_specs.append(
            JobSpec(
                name="gpu_task1",
                script=REPO_DIR / "run_task1_single_gpu.sbatch",
                env={"OUTPUT_DIR": str(gpu_root / "task1")},
            )
        )
    gpu_specs.extend([
        JobSpec(
            name="gpu_task2a",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "gather_scatter",
                "OUTPUT_DIR": str(gpu_root / "task2a"),
                "MASTER_PORT": "29711",
            },
        ),
        JobSpec(
            name="gpu_task2b",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "all_reduce",
                "OUTPUT_DIR": str(gpu_root / "task2b"),
                "MASTER_PORT": "29712",
            },
        ),
        JobSpec(
            name="gpu_task3",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "ddp",
                "OUTPUT_DIR": str(gpu_root / "task3"),
                "MASTER_PORT": "29713",
            },
        ),
        JobSpec(
            name="gpu_task4_gather_scatter",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "gather_scatter",
                "ENABLE_PROFILING": "1",
                "OUTPUT_DIR": str(gpu_profile_root / "gather_scatter" / "run"),
                "PROFILE_OUTPUT_DIR": str(gpu_profile_root / "gather_scatter" / "traces"),
                "MASTER_PORT": "29721",
            },
        ),
        JobSpec(
            name="gpu_task4_all_reduce",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "all_reduce",
                "ENABLE_PROFILING": "1",
                "OUTPUT_DIR": str(gpu_profile_root / "all_reduce" / "run"),
                "PROFILE_OUTPUT_DIR": str(gpu_profile_root / "all_reduce" / "traces"),
                "MASTER_PORT": "29722",
            },
        ),
        JobSpec(
            name="gpu_task4_ddp",
            script=REPO_DIR / "run_task2a_4workers.sbatch",
            env={
                "SYNC_METHOD": "ddp",
                "ENABLE_PROFILING": "1",
                "OUTPUT_DIR": str(gpu_profile_root / "ddp" / "run"),
                "PROFILE_OUTPUT_DIR": str(gpu_profile_root / "ddp" / "traces"),
                "MASTER_PORT": "29723",
            },
        ),
    ])
    return cpu_specs, gpu_specs


def submit_chain(specs: list[JobSpec], extra_sbatch_args: list[str] | None = None) -> tuple[list[dict[str, str]], str]:
    manifest: list[dict[str, str]] = []
    dependency: str | None = None
    for spec in specs:
        job_id = submit_job(spec, dependency=dependency, extra_sbatch_args=extra_sbatch_args)
        manifest.append({"name": spec.name, "job_id": job_id, "script": str(spec.script)})
        dependency = job_id
    return manifest, dependency or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and wait for the full multi-node COS 568 experiment suite.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--skip-build-report", action="store_true")
    parser.add_argument("--skip-task1", action="store_true")
    parser.add_argument("--hardware", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--cpu-account", default="")
    parser.add_argument("--cpu-partition", default="")
    parser.add_argument("--gpu-account", default="")
    parser.add_argument("--gpu-partition", default="")
    parser.add_argument("--cpu-exclude", default="")
    parser.add_argument("--gpu-exclude", default="")
    args = parser.parse_args()

    cpu_specs, gpu_specs = build_job_specs(include_task1=not args.skip_task1)
    cpu_manifest: list[dict[str, str]] = []
    gpu_manifest: list[dict[str, str]] = []
    cpu_final_job = ""
    gpu_final_job = ""
    cpu_extra_args: list[str] = []
    gpu_extra_args: list[str] = []
    if args.cpu_account:
        cpu_extra_args.extend(["-A", args.cpu_account])
    if args.cpu_partition:
        cpu_extra_args.extend(["-p", args.cpu_partition])
    if args.cpu_exclude:
        cpu_extra_args.append(f"--exclude={args.cpu_exclude}")
    if args.gpu_account:
        gpu_extra_args.extend(["-A", args.gpu_account])
    if args.gpu_partition:
        gpu_extra_args.extend(["-p", args.gpu_partition])
    if args.gpu_exclude:
        gpu_extra_args.append(f"--exclude={args.gpu_exclude}")
    if args.hardware in ("cpu", "both"):
        cpu_manifest, cpu_final_job = submit_chain(cpu_specs, extra_sbatch_args=cpu_extra_args)
    if args.hardware in ("gpu", "both"):
        gpu_manifest, gpu_final_job = submit_chain(gpu_specs, extra_sbatch_args=gpu_extra_args)

    manifest = {
        "cpu": cpu_manifest,
        "gpu": gpu_manifest,
        "cpu_final_job": cpu_final_job,
        "gpu_final_job": gpu_final_job,
    }
    manifest_path = REPO_DIR / "outputs" / "report_suite_jobs.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    final_jobs = [job_id for job_id in (cpu_final_job, gpu_final_job) if job_id]
    final_states = wait_for_jobs(final_jobs, poll_seconds=args.poll_seconds)
    failed = {job_id: state for job_id, state in final_states.items() if state != "COMPLETED"}
    if failed:
        raise SystemExit(f"Experiment suite failed: {failed}")

    if not args.skip_build_report:
        subprocess.run([sys.executable, str(REPO_DIR / "scripts" / "build_report.py")], check=True)


if __name__ == "__main__":
    main()
