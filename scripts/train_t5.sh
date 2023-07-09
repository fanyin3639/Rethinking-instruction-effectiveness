#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/local2/fanyin/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port --include "localhost:4" src/run_s2s.py \
    --do_predict \
    --do_train \
    --seed 43 \
    --use_meta_data True \
    --do_analysis False \
    --do_ablation False \
    --annotation_path analysis_util/content_type_annotation.json \
    --label_space_path analysis_util/label_space.json \
    --meta_and_definition False \
    --predict_with_generate \
    --model_name_or_path google/t5-large-lm-adapt \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default/holdout_separate \
    --task_dir data/tasks \
    --output_dir output/t5_large_ablate_additional_info \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 600 \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 10000 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name t5_large_no_additional_info
