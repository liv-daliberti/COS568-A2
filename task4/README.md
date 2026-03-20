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

## Exact Trace JSON Paths Used By The Report

`scripts/build_report.py` reads these exact rank-0 Chrome traces by default:

- `../outputs/task4_cpu_nodes/gather_scatter/traces/rank0_step_window.json`
- `../outputs/task4_cpu_nodes/all_reduce/traces/rank0_step_window.json`
- `../outputs/task4_cpu_nodes/ddp/traces/rank0_step_window.json`
- `../outputs/task4_gpu_nodes/gather_scatter/traces/rank0_step_window.json`
- `../outputs/task4_gpu_nodes/all_reduce/traces/rank0_step_window.json`
- `../outputs/task4_gpu_nodes/ddp/traces/rank0_step_window.json`

Put each exported profiler JSON at the matching `traces/rank0_step_window.json` path above.

For the PDF build:

- Task 4 tables and overview figures are derived from those six JSON trace files.
- Provenance text for the CPU Task 4 setup also reads `run/run_metadata.json` under each method directory.
- Generated report assets land under `../outputs/task4_*_nodes/*/figures` and `../outputs/task4_*_nodes/*/tables`.

## Manual Screenshot Drop Folder

If you want the final PDF to show your own Chrome or Perfetto screenshots, put PNG files here:

- `../report/profiler_images/cpu_gather_scatter.png`
- `../report/profiler_images/cpu_all_reduce.png`
- `../report/profiler_images/cpu_ddp.png`
- `../report/profiler_images/gpu_gather_scatter.png`
- `../report/profiler_images/gpu_all_reduce.png`
- `../report/profiler_images/gpu_ddp.png`

Then rerun:

```bash
./.venv_cos568/bin/python scripts/build_report.py
```
