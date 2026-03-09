# Submission Checklist

## Required Deliverables

- PDF report named exactly: `$NetID$_$firstname$_$lastname$.pdf`
- Source code zip (upload to Canvas)

## Code Layout (already prepared)

- `task1/`
- `task2a/`
- `task2b/`
- `task3/`
- `task4/`

Each task directory contains:

- `run_glue.py`
- `utils_glue.py`

## Task 4 Trace Layout (already prepared)

- `outputs/task4/gather_scatter/traces/rank0_step_window.json`
- `outputs/task4/all_reduce/traces/rank0_step_window.json`
- `outputs/task4/ddp/traces/rank0_step_window.json`

Plus folders for report assets:

- `outputs/task4/{gather_scatter,all_reduce,ddp}/figures/`
- `outputs/task4/{gather_scatter,all_reduce,ddp}/tables/`

## Suggested Packaging Commands

Run from repository root:

```bash
zip -r cos568_a2_code.zip \
  task1 task2a task2b task3 task4 \
  run_task1_single_gpu.sbatch run_task2a_4workers.sbatch run_task2a_4workers_cpu_debug.sbatch \
  REPORT_TEMPLATE.md SUBMISSION_CHECKLIST.md
```

If your final report PDF is ready:

```bash
ls -l $NetID$_$firstname$_$lastname$.pdf cos568_a2_code.zip
```

