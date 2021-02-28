#!/usr/bin/env bash
python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --do_train \
    --train_data_file=./data/finetune_train.txt \
    --do_eval \
    --eval_data_file=./data/finetune_dev.txt \
    --overwrite_output_dir \
    --overwrite_cache \
    --block_size=33 \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --num_train_epochs=1 \
    --save_total_limit=5 \
    --gradient_accumulation_steps=8
    