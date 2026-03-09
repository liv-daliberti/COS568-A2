# Task 1 Code

Single-node fine-tuning for 3 epochs.

## Main Files

- `run_glue.py`
- `utils_glue.py`

## Example Run

```bash
python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name RTE \
  --do_train \
  --do_eval \
  --data_dir ../glue_data/RTE \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ../outputs/task1_rte \
  --overwrite_output_dir
```

