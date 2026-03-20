#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"ProfilerStep#(\d+)$")


@dataclass
class RunSummary:
    avg_iter_time_sec_drop1_mean_over_ranks: float
    num_timed_iters_per_rank: int
    rank_stats: list[dict[str, float | int]]
    loss_curves: dict[int, list[float]]
    nodes: list[str]
    num_nodes: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def latex_graphics_path(path: Path, report_dir: Path) -> str:
    return Path(os.path.relpath(path, start=report_dir)).as_posix()


def paired_png_figure(
    *,
    report_dir: Path,
    left_path: Path,
    right_path: Path,
    left_title: str,
    right_title: str,
    caption: str,
) -> str:
    left_rel = latex_graphics_path(left_path, report_dir)
    right_rel = latex_graphics_path(right_path, report_dir)
    return rf"""\begin{{figure}}[H]
\centering
\begin{{minipage}}[t]{{0.48\linewidth}}
\centering
\textbf{{{latex_escape(left_title)}}}\\[0.3em]
\includegraphics[width=\linewidth]{{{left_rel}}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.48\linewidth}}
\centering
\textbf{{{latex_escape(right_title)}}}\\[0.3em]
\includegraphics[width=\linewidth]{{{right_rel}}}
\end{{minipage}}
\caption{{{latex_escape(caption)}}}
\end{{figure}}"""


def latex_panel_with_optional_image(
    *,
    report_dir: Path,
    image_path: Path,
    title: str,
    placeholder_height: str = "2.7in",
) -> str:
    if image_path.exists():
        image_rel = latex_graphics_path(image_path, report_dir)
        body = rf"\includegraphics[width=\linewidth]{{{image_rel}}}"
    else:
        display_path = latex_escape(str(image_path.relative_to(REPO_DIR)))
        body = rf"""\fbox{{\parbox[c][{placeholder_height}][c]{{0.95\linewidth}}{{\centering
Drop profiler screenshot here\\[0.4em]
\texttt{{{display_path}}}
}}}}"""
    return rf"""\begin{{minipage}}[t]{{0.48\linewidth}}
\centering
\textbf{{{latex_escape(title)}}}\\[0.3em]
{body}
\end{{minipage}}"""


def paired_optional_image_figure(
    *,
    report_dir: Path,
    left_path: Path,
    right_path: Path,
    left_title: str,
    right_title: str,
    caption: str,
    placeholder_height: str = "2.7in",
) -> str:
    left_panel = latex_panel_with_optional_image(
        report_dir=report_dir,
        image_path=left_path,
        title=left_title,
        placeholder_height=placeholder_height,
    )
    right_panel = latex_panel_with_optional_image(
        report_dir=report_dir,
        image_path=right_path,
        title=right_title,
        placeholder_height=placeholder_height,
    )
    return rf"""\begin{{figure}}[H]
\centering
{left_panel}\hfill
{right_panel}
\caption{{{latex_escape(caption)}}}
\end{{figure}}"""


def load_rank_metrics(run_dir: Path) -> list[dict]:
    metrics_dir = run_dir / "metrics"
    metrics = [read_json(path) for path in sorted(metrics_dir.glob("rank*_train_metrics.json"))]
    if not metrics:
        raise FileNotFoundError(f"No metrics found in {metrics_dir}")
    return sorted(metrics, key=lambda item: item["rank"])


def load_run_metadata(path: Path) -> dict:
    if not path.exists():
        return {"nodes": [], "num_nodes": 0}
    return read_json(path)


def write_loss_csv(run_dir: Path, metrics: list[dict], output_path: Path) -> None:
    max_len = max(len(metric["step_losses"]) for metric in metrics)
    fieldnames = ["step"] + [f"rank{metric['rank']}" for metric in metrics]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step_idx in range(max_len):
            row = {"step": step_idx + 1}
            for metric in metrics:
                losses = metric["step_losses"]
                row[f"rank{metric['rank']}"] = losses[step_idx] if step_idx < len(losses) else ""
            writer.writerow(row)


def plot_loss_curves(metrics: list[dict], output_path: Path, title: str) -> dict[int, list[float]]:
    plt.figure(figsize=(8, 4.5))
    loss_curves: dict[int, list[float]] = {}
    for metric in metrics:
        rank = int(metric["rank"])
        losses = list(metric["step_losses"])
        loss_curves[rank] = losses
        plt.plot(range(1, len(losses) + 1), losses, label=f"rank {rank}", linewidth=1.8)
    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return loss_curves


def summarize_distributed_run(run_dir: Path, loss_csv_path: Path, loss_plot_path: Path, title: str) -> RunSummary:
    metrics = load_rank_metrics(run_dir)
    metadata = load_run_metadata(run_dir / "run_metadata.json")
    write_loss_csv(run_dir, metrics, loss_csv_path)
    loss_curves = plot_loss_curves(metrics, loss_plot_path, title)

    rank_stats = []
    rank_means = []
    timed_iters = None
    for metric in metrics:
        step_times = list(metric["step_times_sec"])
        if len(step_times) <= 1:
            raise ValueError(f"Expected more than one step time in {run_dir}")
        trimmed = step_times[1:]
        avg_time = sum(trimmed) / len(trimmed)
        timed_iters = len(trimmed)
        rank_stats.append(
            {
                "rank": int(metric["rank"]),
                "avg_iter_time_sec_drop1": avg_time,
                "num_timed_iters": timed_iters,
            }
        )
        rank_means.append(avg_time)

    return RunSummary(
        avg_iter_time_sec_drop1_mean_over_ranks=sum(rank_means) / len(rank_means),
        num_timed_iters_per_rank=timed_iters or 0,
        rank_stats=rank_stats,
        loss_curves=loss_curves,
        nodes=list(metadata.get("nodes", [])),
        num_nodes=int(metadata.get("num_nodes", 0)),
    )


def max_abs_loss_diff(loss_curves_a: dict[int, list[float]], loss_curves_b: dict[int, list[float]]) -> tuple[dict[int, float], float]:
    per_rank: dict[int, float] = {}
    global_max = 0.0
    for rank in sorted(loss_curves_a):
        diffs = [
            abs(a - b)
            for a, b in zip(loss_curves_a[rank], loss_curves_b.get(rank, []), strict=False)
        ]
        per_rank_max = max(diffs) if diffs else 0.0
        per_rank[rank] = per_rank_max
        global_max = max(global_max, per_rank_max)
    return per_rank, global_max


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return []
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def load_trace_events(trace_path: Path) -> list[dict]:
    trace = read_json(trace_path)
    if isinstance(trace, dict):
        return trace.get("traceEvents", [])
    return trace


def extract_step_windows(events: list[dict]) -> list[dict]:
    windows: dict[int, dict] = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        match = STEP_RE.match(name)
        if match is None:
            continue
        step_id = int(match.group(1))
        if step_id not in windows or float(event.get("dur", 0.0)) > float(windows[step_id].get("dur", 0.0)):
            windows[step_id] = event
    return [windows[key] for key in sorted(windows)]


def analyze_trace(trace_path: Path) -> tuple[list[dict], list[list[tuple[float, float]]]]:
    events = load_trace_events(trace_path)
    step_windows = extract_step_windows(events)
    if not step_windows:
        raise ValueError(f"No ProfilerStep windows found in {trace_path}")

    comm_events = [
        event
        for event in events
        if event.get("ph") == "X" and str(event.get("name", "")).startswith(("gloo:", "nccl:"))
    ]

    rows = []
    merged_intervals_per_step: list[list[tuple[float, float]]] = []
    for step_idx, step_event in enumerate(step_windows, start=1):
        step_start = float(step_event["ts"])
        step_end = step_start + float(step_event["dur"])
        intervals: list[tuple[float, float]] = []
        for event in comm_events:
            event_start = float(event["ts"])
            event_end = event_start + float(event["dur"])
            overlap_start = max(step_start, event_start)
            overlap_end = min(step_end, event_end)
            if overlap_end > overlap_start:
                intervals.append((overlap_start, overlap_end))
        merged = merge_intervals(intervals)
        merged_intervals_per_step.append(merged)
        comm_us = sum(end - start for start, end in merged)
        total_ms = float(step_event["dur"]) / 1000.0
        comm_ms = comm_us / 1000.0
        rows.append(
            {
                "step": step_idx,
                "total_ms": total_ms,
                "comm_ms": comm_ms,
                "comm_overhead_pct": (comm_ms / total_ms) * 100.0 if total_ms else 0.0,
            }
        )
    return rows, merged_intervals_per_step


def write_comm_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "total_ms", "comm_ms", "comm_overhead_pct"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_trace_overview(
    trace_path: Path,
    rows: list[dict],
    merged_intervals_per_step: list[list[tuple[float, float]]],
    output_path: Path,
    title: str,
) -> None:
    events = load_trace_events(trace_path)
    step_windows = extract_step_windows(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    y_positions = list(range(len(step_windows), 0, -1))
    for idx, (step_event, merged) in enumerate(zip(step_windows, merged_intervals_per_step, strict=True)):
        y = y_positions[idx]
        step_start = float(step_event["ts"])
        step_ms = float(step_event["dur"]) / 1000.0
        ax.broken_barh([(0.0, step_ms)], (y - 0.35, 0.7), facecolors="#d9d9d9")
        comm_segments = [((start - step_start) / 1000.0, (end - start) / 1000.0) for start, end in merged]
        if comm_segments:
            ax.broken_barh(comm_segments, (y - 0.35, 0.7), facecolors="#2b6cb0")
        ax.text(
            step_ms + max(row["total_ms"] for row in rows) * 0.02,
            y,
            f"{rows[idx]['comm_overhead_pct']:.2f}%",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"step {row['step']}" for row in rows])
    ax.set_xlabel("Milliseconds within profiled step")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_profile_method(method_dir: Path, title: str) -> dict:
    trace_path = method_dir / "traces" / "rank0_step_window.json"
    rows, merged_intervals_per_step = analyze_trace(trace_path)
    write_comm_csv(rows, method_dir / "tables" / "comm_overhead_per_step.csv")
    plot_trace_overview(
        trace_path=trace_path,
        rows=rows,
        merged_intervals_per_step=merged_intervals_per_step,
        output_path=method_dir / "figures" / "rank0_steps1to3_overview.png",
        title=title,
    )
    return {
        "rows": rows,
        "avg_comm_overhead_pct": sum(row["comm_overhead_pct"] for row in rows) / len(rows),
    }


def build_timing_summary(root_dir: Path, hardware_label: str) -> dict:
    task_specs = {
        "task2a": ("task2a", "Task 2(a) loss curve", root_dir / "task2a_losses_per_rank.csv", root_dir / "task2a" / "figures" / "loss_curve_all_ranks.png"),
        "task2b": ("task2b", "Task 2(b) loss curve", root_dir / "task2b_losses_per_rank.csv", root_dir / "task2b" / "figures" / "loss_curve_all_ranks.png"),
        "task3": ("task3", "Task 3 loss curve", root_dir / "task3_losses_per_rank.csv", root_dir / "task3" / "figures" / "loss_curve_all_ranks.png"),
    }

    summary: dict[str, object] = {}
    run_summaries: dict[str, RunSummary] = {}
    for task_key, (run_name, plot_title, loss_csv, loss_plot) in task_specs.items():
        run_summary = summarize_distributed_run(
            run_dir=root_dir / run_name,
            loss_csv_path=loss_csv,
            loss_plot_path=loss_plot,
            title=f"{hardware_label} {plot_title}",
        )
        run_summaries[task_key] = run_summary
        summary[task_key] = {
            "rank_stats": run_summary.rank_stats,
            "avg_iter_time_sec_drop1_mean_over_ranks": run_summary.avg_iter_time_sec_drop1_mean_over_ranks,
            "num_timed_iters_per_rank": run_summary.num_timed_iters_per_rank,
            "nodes": run_summary.nodes,
            "num_nodes": run_summary.num_nodes,
        }

    per_rank_diff, global_diff = max_abs_loss_diff(run_summaries["task2a"].loss_curves, run_summaries["task2b"].loss_curves)
    summary["task2a_vs_task2b_loss_max_abs_diff_per_rank"] = {str(rank): value for rank, value in per_rank_diff.items()}
    summary["task2a_vs_task2b_loss_max_abs_diff_global"] = global_diff

    (root_dir / "timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root_dir / "timing_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "avg_iter_time_sec_drop1_mean_over_ranks", "num_timed_iters_per_rank", "num_nodes"])
        for task_key in ("task2a", "task2b", "task3"):
            task_summary = summary[task_key]
            writer.writerow(
                [
                    task_key,
                    task_summary["avg_iter_time_sec_drop1_mean_over_ranks"],
                    task_summary["num_timed_iters_per_rank"],
                    task_summary["num_nodes"],
                ]
            )
    return summary


def build_profile_summary(root_dir: Path, hardware_label: str) -> dict:
    titles = {
        "gather_scatter": f"{hardware_label} gather/scatter rank 0 profiled window",
        "all_reduce": f"{hardware_label} all-reduce rank 0 profiled window",
        "ddp": f"{hardware_label} DDP rank 0 profiled window",
    }
    summary = {
        method: summarize_profile_method(root_dir / method, title)
        for method, title in titles.items()
    }
    all_reduce_avg = summary["all_reduce"]["avg_comm_overhead_pct"]
    ddp_avg = summary["ddp"]["avg_comm_overhead_pct"]
    summary["ddp_vs_all_reduce_relative_overhead_reduction_pct"] = (
        ((all_reduce_avg - ddp_avg) / all_reduce_avg) * 100.0 if all_reduce_avg else 0.0
    )
    summary["ddp_vs_all_reduce_comm_share_delta_pct_points"] = ddp_avg - all_reduce_avg
    summary["ddp_vs_all_reduce_comm_share_relative_change_pct"] = (
        ((ddp_avg - all_reduce_avg) / all_reduce_avg) * 100.0 if all_reduce_avg else 0.0
    )
    (root_dir / "comm_overhead_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def summarize_task1(run_dir: Path) -> dict:
    metrics = load_rank_metrics(run_dir)
    if len(metrics) != 1:
        raise ValueError(f"Expected single-rank Task 1 metrics in {run_dir}")
    metric = metrics[0]
    metadata = load_run_metadata(run_dir / "run_metadata.json")
    epoch_metrics = metric["epoch_eval_results"]
    return {
        "epoch_acc": [entry["metrics"]["acc"] for entry in epoch_metrics],
        "first5_minibatch_loss": metric["step_losses"][:5],
        "nodes": metadata.get("nodes", []),
        "num_nodes": metadata.get("num_nodes", 0),
    }


def git_short_hash(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def render_report(
    report_tex_path: Path,
    cpu_root: Path,
    gpu_root: Path,
    cpu_profile_root: Path,
    gpu_profile_root: Path,
    cpu_task1: dict,
    gpu_task1: dict,
    cpu_timing: dict,
    gpu_timing: dict,
    cpu_profile: dict,
    gpu_profile: dict,
    build_time: datetime,
) -> None:
    def pct_improvement(baseline: float, candidate: float) -> float:
        return ((baseline - candidate) / baseline) * 100.0 if baseline else 0.0

    def latex_rel(path: Path) -> str:
        return latex_escape(str(path.relative_to(REPO_DIR)))

    timing_rows = []
    for hardware, timing_summary in (("CPU", cpu_timing), ("GPU", gpu_timing)):
        timing_rows.extend(
            [
                (hardware, "2(a)", "gather+scatter", timing_summary["task2a"]),
                (hardware, "2(b)", "all_reduce", timing_summary["task2b"]),
                (hardware, "3", "DDP", timing_summary["task3"]),
            ]
        )

    cpu_2a = cpu_timing["task2a"]["avg_iter_time_sec_drop1_mean_over_ranks"]
    cpu_2b = cpu_timing["task2b"]["avg_iter_time_sec_drop1_mean_over_ranks"]
    cpu_3 = cpu_timing["task3"]["avg_iter_time_sec_drop1_mean_over_ranks"]
    gpu_2a = gpu_timing["task2a"]["avg_iter_time_sec_drop1_mean_over_ranks"]
    gpu_2b = gpu_timing["task2b"]["avg_iter_time_sec_drop1_mean_over_ranks"]
    gpu_3 = gpu_timing["task3"]["avg_iter_time_sec_drop1_mean_over_ranks"]

    cpu_allreduce_gain = pct_improvement(cpu_2a, cpu_2b)
    cpu_ddp_gain = pct_improvement(cpu_2b, cpu_3)
    gpu_allreduce_gain = pct_improvement(gpu_2a, gpu_2b)
    gpu_ddp_gain = pct_improvement(gpu_2b, gpu_3)
    gpu_speedups = {
        "2a": cpu_2a / gpu_2a,
        "2b": cpu_2b / gpu_2b,
        "3": cpu_3 / gpu_3,
    }
    cpu_task2a_nodes = cpu_timing["task2a"]["nodes"]
    cpu_task2b_nodes = cpu_timing["task2b"]["nodes"]
    cpu_task3_nodes = cpu_timing["task3"]["nodes"]
    cpu_profiles_nodes = read_json(cpu_profile_root / "all_reduce" / "run" / "run_metadata.json").get("nodes", [])
    cpu_nodes_are_uniform = cpu_task2a_nodes == cpu_task2b_nodes == cpu_task3_nodes
    cpu_timing_nodes_line = (
        rf"CPU distributed timing runs: Tasks 2(a), 2(b), and 3 on \texttt{{{latex_escape(', '.join(cpu_task2a_nodes))}}}; 1 rank per node, backend \texttt{{gloo}}."
        if cpu_nodes_are_uniform
        else rf"CPU distributed timing runs: Task 2(a) on \texttt{{{latex_escape(', '.join(cpu_task2a_nodes))}}}; Tasks 2(b)/3 on \texttt{{{latex_escape(', '.join(cpu_task2b_nodes))}}}; 1 rank per node, backend \texttt{{gloo}}."
    )
    timing_rows_latex = "\n".join(
        f"{hardware} & {task} & {latex_escape(method)} & "
        f"{summary['avg_iter_time_sec_drop1_mean_over_ranks']:.6f} & "
        f"{summary['num_timed_iters_per_rank']} \\\\"
        for hardware, task, method, summary in timing_rows
    )
    cpu_profile_rows_latex = "\n".join(
        f"CPU & {latex_escape(method)} & {row['step']} & {row['total_ms']:.3f} & "
        f"{row['comm_ms']:.3f} / {row['comm_overhead_pct']:.3f}\\% \\\\"
        for method in ("gather_scatter", "all_reduce", "ddp")
        for row in cpu_profile[method]["rows"]
    )
    gpu_profile_rows_latex = "\n".join(
        f"GPU & {latex_escape(method)} & {row['step']} & {row['total_ms']:.3f} & "
        f"{row['comm_ms']:.3f} / {row['comm_overhead_pct']:.3f}\\% \\\\"
        for method in ("gather_scatter", "all_reduce", "ddp")
        for row in gpu_profile[method]["rows"]
    )
    task2_loss_figures_latex = "\n\n".join(
        paired_png_figure(
            report_dir=report_tex_path.parent,
            left_path=cpu_root / task_dir / "figures" / "loss_curve_all_ranks.png",
            right_path=gpu_root / task_dir / "figures" / "loss_curve_all_ranks.png",
            left_title=f"CPU {task_label}",
            right_title=f"GPU {task_label}",
            caption=f"{task_label} loss curves across all ranks.",
        )
        for task_dir, task_label in (
            ("task2a", "Task 2(a)"),
            ("task2b", "Task 2(b)"),
            ("task3", "Task 3"),
        )
    )
    task4_profile_figures_latex = "\n\n".join(
        paired_png_figure(
            report_dir=report_tex_path.parent,
            left_path=cpu_profile_root / method / "figures" / "rank0_steps1to3_overview.png",
            right_path=gpu_profile_root / method / "figures" / "rank0_steps1to3_overview.png",
            left_title=f"CPU {method.replace('_', ' ')}",
            right_title=f"GPU {method.replace('_', ' ')}",
            caption=f"Profiler overview for {method.replace('_', ' ')} over profiled steps 1-3.",
        )
        for method in ("gather_scatter", "all_reduce", "ddp")
    )
    manual_profiler_image_root = REPO_DIR / "report" / "profiler_images"
    manual_profiler_figures_latex = "\n\n".join(
        paired_optional_image_figure(
            report_dir=report_tex_path.parent,
            left_path=manual_profiler_image_root / f"cpu_{method}.png",
            right_path=manual_profiler_image_root / f"gpu_{method}.png",
            left_title=f"CPU {method.replace('_', ' ')} Chrome trace",
            right_title=f"GPU {method.replace('_', ' ')} Chrome trace",
            caption=f"Chrome trace screenshot for {method.replace('_', ' ')}.",
            placeholder_height="2.9in",
        )
        for method in ("gather_scatter", "all_reduce", "ddp")
    )

    report_tex_path.parent.mkdir(parents=True, exist_ok=True)
    report_tex_path.write_text(
        rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\title{{COS 568 Assignment 2: Distributed Training of Language Models}}
\author{{NetID: od2961 \quad Liv d'Aliberti}}
\date{{{build_time.strftime('%B %d, %Y')}}}

\begin{{document}}
\maketitle

\section{{Setup and Reproducibility}}
\begin{{itemize}}
\item Task 1 CPU run: {cpu_task1["num_nodes"]} node ({latex_escape(", ".join(cpu_task1["nodes"]))}), backend \texttt{{gloo}}-free single-process training.
\item Task 1 GPU run: {gpu_task1["num_nodes"]} node ({latex_escape(", ".join(gpu_task1["nodes"]))}), single GPU.
\item {cpu_timing_nodes_line}
\item CPU Task 4 profiling runs: \texttt{{{latex_escape(", ".join(cpu_profiles_nodes))}}}; 1 rank per node, backend \texttt{{gloo}}.
\item Multi-node GPU distributed runs: {gpu_timing["task2a"]["num_nodes"]} nodes, {gpu_timing["task2a"]["num_nodes"]} ranks, 1 rank per node, backend \texttt{{nccl}}.
\item GPU distributed nodes: \texttt{{{latex_escape(", ".join(gpu_timing["task2a"]["nodes"]))}}}
\item Seed: \texttt{{42}}, PyTorch version: \texttt{{{latex_escape(torch.__version__)}}}, commit: \texttt{{{latex_escape(git_short_hash(REPO_DIR))}}}.
\end{{itemize}}

\section{{Task 1: Single-Node Fine-Tuning}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{ccc}}
\toprule
Epoch & CPU acc & GPU acc \\
\midrule
1 & {cpu_task1["epoch_acc"][0]:.10f} & {gpu_task1["epoch_acc"][0]:.10f} \\
2 & {cpu_task1["epoch_acc"][1]:.10f} & {gpu_task1["epoch_acc"][1]:.10f} \\
3 & {cpu_task1["epoch_acc"][2]:.10f} & {gpu_task1["epoch_acc"][2]:.10f} \\
\bottomrule
\end{{tabular}}
\caption{{Task 1 evaluation accuracy after each epoch.}}
\end{{table}}

\begin{{table}}[H]
\centering
\begin{{tabular}}{{ccc}}
\toprule
Minibatch & CPU loss & GPU loss \\
\midrule
1 & {cpu_task1["first5_minibatch_loss"][0]:.6f} & {gpu_task1["first5_minibatch_loss"][0]:.6f} \\
2 & {cpu_task1["first5_minibatch_loss"][1]:.6f} & {gpu_task1["first5_minibatch_loss"][1]:.6f} \\
3 & {cpu_task1["first5_minibatch_loss"][2]:.6f} & {gpu_task1["first5_minibatch_loss"][2]:.6f} \\
4 & {cpu_task1["first5_minibatch_loss"][3]:.6f} & {gpu_task1["first5_minibatch_loss"][3]:.6f} \\
5 & {cpu_task1["first5_minibatch_loss"][4]:.6f} & {gpu_task1["first5_minibatch_loss"][4]:.6f} \\
\bottomrule
\end{{tabular}}
\caption{{Task 1 first five minibatch losses.}}
\end{{table}}

\section{{Tasks 2(a), 2(b), and 3}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Hardware & Task & Method & Avg iter time (s) & Timed iters \\
\midrule
{timing_rows_latex}
\bottomrule
\end{{tabular}}
\caption{{Average iteration time after dropping the first iteration.}}
\end{{table}}

CPU Task 2(a) vs 2(b) max per-step loss difference: {cpu_timing["task2a_vs_task2b_loss_max_abs_diff_global"]:.3e}.\\
GPU Task 2(a) vs 2(b) max per-step loss difference: {gpu_timing["task2a_vs_task2b_loss_max_abs_diff_global"]:.3e}.

{task2_loss_figures_latex}

\section{{Task 4: Profiling and Communication Overhead}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Hardware & Method & Step & Total ms & Comm ms / pct \\
\midrule
{cpu_profile_rows_latex}
{gpu_profile_rows_latex}
\bottomrule
\end{{tabular}}
\caption{{Per-step communication overhead from the profiled three-step windows.}}
\end{{table}}

{task4_profile_figures_latex}

\subsection{{Chrome Trace Screenshots}}

{manual_profiler_figures_latex}

\section{{Implementation Details}}
The code is organized under \texttt{{task1}}, \texttt{{task2a}}, \texttt{{task2b}}, \texttt{{task3}}, and \texttt{{task4}}, matching the deliverable layout requested in the assignment. All distributed runs keep the same total batch size as Task 1 by using per-rank batch size 16 across 4 workers.

Task 1 implements the standard single-node training loop with backward pass, optimizer step, and per-epoch evaluation. Task 2(a) uses a \texttt{{DistributedSampler}} and manual gradient averaging with \texttt{{gather}} followed by \texttt{{scatter}} through rank 0. Task 2(b) replaces that centralized synchronization with \texttt{{all\_reduce}} and divides by world size on every rank. Task 3 wraps the model in \texttt{{DistributedDataParallel}}, leaving gradient bucketing and synchronization to PyTorch.

Task 4 reuses the same training code with \texttt{{torch.profiler}} enabled. The profiler schedule skips the first training step and then records the next three steps (\texttt{{wait=1, warmup=0, active=3}}), exporting a rank-0 Chrome trace JSON for each communication method.

\section{{Discussion}}
Across both CPU and GPU, the qualitative ordering is consistent: \texttt{{gather}}/\texttt{{scatter}} is slowest, \texttt{{all\_reduce}} is faster, and \texttt{{DistributedDataParallel}} is fastest. On CPU, Task 2(b) is {cpu_allreduce_gain:.2f}\% faster than Task 2(a), and Task 3 is another {cpu_ddp_gain:.2f}\% faster than Task 2(b). On GPU, Task 2(b) is {gpu_allreduce_gain:.2f}\% faster than Task 2(a), and Task 3 is another {gpu_ddp_gain:.2f}\% faster than Task 2(b). The GPU runs are still much faster overall, with GPU/CPU speedups of {gpu_speedups["2a"]:.2f}x for Task 2(a), {gpu_speedups["2b"]:.2f}x for Task 2(b), and {gpu_speedups["3"]:.2f}x for Task 3.

The Task 2(a) and Task 2(b) loss curves match to within {cpu_timing["task2a_vs_task2b_loss_max_abs_diff_global"]:.3e} on CPU and {gpu_timing["task2a_vs_task2b_loss_max_abs_diff_global"]:.3e} on GPU, which is effectively identical at floating-point precision. That confirms the two manual synchronization implementations are producing the same training trajectory while differing primarily in communication cost.

The lack of difference between Task 2(a) and Task 2(b) in the loss curves is expected. As described in the PyTorch Distributed VLDB'18 paper, distributed data parallel training is designed to be mathematically equivalent to local training when replicas start from the same parameters and synchronize gradients each iteration. In this setup, Tasks 2(a), 2(b), and 3 use the same total batch size and seed, so the main difference among them is communication strategy and runtime cost rather than optimization behavior.

The CPU and GPU runs differ mainly in hardware throughput and backend efficiency rather than in training semantics: GPUs reduce both computation time and collective latency, so the same communication methods preserve ordering but run much faster.

Relative to manual all-reduce, DDP cuts average iteration time by {cpu_ddp_gain:.2f}\% on CPU and {gpu_ddp_gain:.2f}\% on GPU. That end-to-end reduction is the clearest efficiency comparison for Task 4; the corresponding all-reduce gains over gather/scatter are {cpu_allreduce_gain:.2f}\% on CPU and {gpu_allreduce_gain:.2f}\% on GPU.

The Task 4 communication-share metric is computed on rank 0 as the union of all \texttt{{gloo:}}/\texttt{{nccl:}} intervals inside each \texttt{{ProfilerStep}} window. Under that metric, CPU DDP averages {cpu_profile["ddp"]["avg_comm_overhead_pct"]:.3f}\% versus {cpu_profile["all_reduce"]["avg_comm_overhead_pct"]:.3f}\% for manual all-reduce ({cpu_profile["ddp_vs_all_reduce_comm_share_delta_pct_points"]:+.3f} percentage points), and GPU DDP averages {gpu_profile["ddp"]["avg_comm_overhead_pct"]:.3f}\% versus {gpu_profile["all_reduce"]["avg_comm_overhead_pct"]:.3f}\% ({gpu_profile["ddp_vs_all_reduce_comm_share_delta_pct_points"]:+.3f} points). This does not contradict DDP's better end-to-end time: DDP overlaps reduction with backpropagation and uses optimized gradient buckets, so a larger fraction of a shorter step can still be communication-tagged. This is consistent with the PyTorch Distributed design described in the VLDB'18 paper.

These results also illustrate a practical scalability limit for distributed ML. Better collectives clearly improve throughput, but synchronization still occupies a large share of each profiled step, especially once computation becomes faster. In other words, scaling improves when communication is decentralized and overlapped, but communication overhead does not disappear; it becomes the main factor that limits how close distributed training gets to ideal linear speedup.

\end{{document}}
""",
        encoding="utf-8",
    )


def compile_report(report_tex_path: Path) -> None:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        report_tex_path.name,
    ]
    for _ in range(2):
        subprocess.run(command, cwd=report_tex_path.parent, check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fresh summaries, figures, and the updated COS 568 report.")
    parser.add_argument("--cpu-root", type=Path, default=REPO_DIR / "outputs" / "report_runs_cpu_nodes")
    parser.add_argument("--gpu-root", type=Path, default=REPO_DIR / "outputs" / "report_runs_gpu_nodes")
    parser.add_argument("--cpu-profile-root", type=Path, default=REPO_DIR / "outputs" / "task4_cpu_nodes")
    parser.add_argument("--gpu-profile-root", type=Path, default=REPO_DIR / "outputs" / "task4_gpu_nodes")
    parser.add_argument("--report-tex", type=Path, default=REPO_DIR / "report" / "od2961_liv_daliberti.tex")
    parser.add_argument("--report-pdf", type=Path, default=REPO_DIR / "report" / "od2961_liv_daliberti.pdf")
    args = parser.parse_args()
    args.cpu_root = args.cpu_root.resolve()
    args.gpu_root = args.gpu_root.resolve()
    args.cpu_profile_root = args.cpu_profile_root.resolve()
    args.gpu_profile_root = args.gpu_profile_root.resolve()
    args.report_tex = args.report_tex.resolve()
    args.report_pdf = args.report_pdf.resolve()

    cpu_task1 = summarize_task1(args.cpu_root / "task1")
    gpu_task1 = summarize_task1(args.gpu_root / "task1")
    cpu_timing = build_timing_summary(args.cpu_root, "CPU")
    gpu_timing = build_timing_summary(args.gpu_root, "GPU")
    cpu_profile = build_profile_summary(args.cpu_profile_root, "CPU")
    gpu_profile = build_profile_summary(args.gpu_profile_root, "GPU")

    render_report(
        report_tex_path=args.report_tex,
        cpu_root=args.cpu_root,
        gpu_root=args.gpu_root,
        cpu_profile_root=args.cpu_profile_root,
        gpu_profile_root=args.gpu_profile_root,
        cpu_task1=cpu_task1,
        gpu_task1=gpu_task1,
        cpu_timing=cpu_timing,
        gpu_timing=gpu_timing,
        cpu_profile=cpu_profile,
        gpu_profile=gpu_profile,
        build_time=datetime.now(),
    )
    compile_report(args.report_tex)

    built_pdf = args.report_tex.with_suffix(".pdf")
    if built_pdf != args.report_pdf:
        args.report_pdf.write_bytes(built_pdf.read_bytes())


if __name__ == "__main__":
    main()
