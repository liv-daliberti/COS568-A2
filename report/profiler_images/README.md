# Manual Profiler Screenshot Slots

Drop manual profiler screenshots here as PNG files, then rerun:

```bash
./.venv_cos568/bin/python scripts/build_report.py
```

Expected filenames:

- `cpu_gather_scatter.png`
- `cpu_all_reduce.png`
- `cpu_ddp.png`
- `gpu_gather_scatter.png`
- `gpu_all_reduce.png`
- `gpu_ddp.png`

Source trace JSON files to open in Chrome tracing or Perfetto:

- `../../outputs/task4_cpu_nodes/gather_scatter/traces/rank0_step_window.json`
- `../../outputs/task4_cpu_nodes/all_reduce/traces/rank0_step_window.json`
- `../../outputs/task4_cpu_nodes/ddp/traces/rank0_step_window.json`
- `../../outputs/task4_gpu_nodes/gather_scatter/traces/rank0_step_window.json`
- `../../outputs/task4_gpu_nodes/all_reduce/traces/rank0_step_window.json`
- `../../outputs/task4_gpu_nodes/ddp/traces/rank0_step_window.json`
