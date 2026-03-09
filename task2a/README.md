# Task 2(a) Code

Distributed training with manual `gather`/`scatter` gradient synchronization.

## Main Files

- `run_glue.py`
- `utils_glue.py`

## Required Distributed CLI Shape

```bash
python run_glue.py [other args] \
  --master_ip $ip_address$ \
  --master_port $port$ \
  --world_size 4 \
  --local_rank $rank$ \
  --sync_method gather_scatter
```

