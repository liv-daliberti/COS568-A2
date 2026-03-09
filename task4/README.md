# Task 4 Code and Artifacts

Profiling instrumentation is in `run_glue.py` using `torch.profiler`.

## Main Files

- `run_glue.py`
- `utils_glue.py`

## Profiling Flags

```bash
--enable_profiling \
--profile_output_dir <path> \
--profile_wait_steps 1 \
--profile_active_steps 3
```

## Current Trace Paths

- `../outputs/task4/gather_scatter/traces/rank0_step_window.json`
- `../outputs/task4/all_reduce/traces/rank0_step_window.json`
- `../outputs/task4/ddp/traces/rank0_step_window.json`

Use `../outputs/task4/*/figures` and `../outputs/task4/*/tables` for report assets.

