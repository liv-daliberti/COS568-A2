# COS 568 A2 Report Template

> Rename final PDF to: `$NetID$_$firstname$_$lastname$.pdf`
> Example: `od2961_oliver_doe.pdf`

## 1. Setup and Reproducibility

- NetID: `[FILL]`
- Name: `[FILL]`
- Date: `[FILL]`
- Repo commit hash: `[FILL]`
- Environment:
  - Nodes: `[FILL]`
  - CPU/GPU: `[FILL]`
  - Backend: `[gloo | nccl]`
  - PyTorch version: `[FILL]`
- Fixed random seed used: `[FILL]`

## 2. Task 1 (Single-Node Fine-Tuning, 3 Epochs)

### 2.1 Training Configuration

- Model: `bert-base-cased`
- Dataset: `GLUE/RTE`
- Epochs: `3`
- Total batch size: `64`
- Learning rate: `2e-5`
- Other details: `[FILL if changed]`

### 2.2 Required Result: Eval Metric After Every Epoch

| Epoch | Eval Metric (`acc`) | Notes |
|---|---:|---|
| 1 | `[FILL]` | `[FILL]` |
| 2 | `[FILL]` | `[FILL]` |
| 3 | `[FILL]` | `[FILL]` |

### 2.3 First Five Minibatch Losses

| Minibatch | Loss |
|---|---:|
| 1 | `[FILL]` |
| 2 | `[FILL]` |
| 3 | `[FILL]` |
| 4 | `[FILL]` |
| 5 | `[FILL]` |

## 3. Task 2(a) and 2(b), Task 3 (1 Epoch Each)

### 3.1 Timing Method

- Each task runs for `1` epoch.
- Discarded timing for first iteration: `Yes`
- Average computed over remaining iterations: `Yes`
- Timing API used: `[FILL, e.g., time.perf_counter]`

### 3.2 Average Iteration Time (After Dropping Iteration 1)

| Task | Comm Method | Avg Iter Time (s) | Num Timed Iters | Notes |
|---|---|---:|---:|---|
| 2(a) | gather+scatter | `[FILL]` | `[FILL]` | `[FILL]` |
| 2(b) | all_reduce | `[FILL]` | `[FILL]` | `[FILL]` |
| 3 | DDP | `[FILL]` | `[FILL]` | `[FILL]` |

### 3.3 Loss Curves Per Node

- Figure slot (Task 2(a) per-node loss): `[INSERT FIGURE]`
- Figure slot (Task 2(b) per-node loss): `[INSERT FIGURE]`
- Figure slot (Task 3 per-node loss): `[INSERT FIGURE]`
- Consistency check 2(a) vs 2(b): `[FILL: same or explain differences]`

## 4. Task 4 Profiling (3 Steps, Skip First Step)

### 4.1 Trace Files

- gather/scatter trace:
  - `outputs/task4/gather_scatter/traces/rank0_step_window.json`
- all_reduce trace:
  - `outputs/task4/all_reduce/traces/rank0_step_window.json`
- DDP trace:
  - `outputs/task4/ddp/traces/rank0_step_window.json`

### 4.2 Communication Overhead Per Step

Fill one table per method (3 profiled steps each).

#### gather/scatter

| Step | Total Time (ms) | Comm Time (ms) | Comm Overhead (%) |
|---|---:|---:|---:|
| 1 | `[FILL]` | `[FILL]` | `[FILL]` |
| 2 | `[FILL]` | `[FILL]` | `[FILL]` |
| 3 | `[FILL]` | `[FILL]` | `[FILL]` |

#### all_reduce

| Step | Total Time (ms) | Comm Time (ms) | Comm Overhead (%) |
|---|---:|---:|---:|
| 1 | `[FILL]` | `[FILL]` | `[FILL]` |
| 2 | `[FILL]` | `[FILL]` | `[FILL]` |
| 3 | `[FILL]` | `[FILL]` | `[FILL]` |

#### DDP

| Step | Total Time (ms) | Comm Time (ms) | Comm Overhead (%) |
|---|---:|---:|---:|
| 1 | `[FILL]` | `[FILL]` | `[FILL]` |
| 2 | `[FILL]` | `[FILL]` | `[FILL]` |
| 3 | `[FILL]` | `[FILL]` | `[FILL]` |

### 4.3 all_reduce vs DDP Comparison

- DDP average communication overhead: `[FILL]%`
- all_reduce average communication overhead: `[FILL]%`
- Relative reduction of DDP vs all_reduce:
  - `((all_reduce - ddp) / all_reduce) * 100 = [FILL]%`
- Why DDP is more efficient in your run:
  - `[FILL: gradient bucketing, overlap of comm/compute, optimized internals, etc.]`

## 5. Discussion

### 5.1 Difference Across Setups

- Observed differences (or no differences): `[FILL]`
- Explanation: `[FILL]`

### 5.2 Scalability of Distributed ML

- Trend with 1 -> 4 workers: `[FILL]`
- Main bottlenecks: `[FILL]`
- Practical implications: `[FILL]`

### 5.3 Relation to PyTorch Distributed (VLDB'18)

- Key concept(s) used to interpret results: `[FILL]`
- Link from concept to your measurements: `[FILL]`

## 6. Implementation Details

- Task 1 implementation summary: `[FILL]`
- Task 2(a) implementation summary: `[FILL]`
- Task 2(b) implementation summary: `[FILL]`
- Task 3 implementation summary: `[FILL]`
- Task 4 profiling instrumentation summary: `[FILL]`
- Any debugging notes or caveats: `[FILL]`

## 7. Appendix (Optional)

- Extra plots: `[INSERT]`
- Additional logs/tables: `[INSERT]`

