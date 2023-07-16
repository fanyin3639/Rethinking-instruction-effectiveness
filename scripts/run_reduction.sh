#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH=/path/to/Rethinking-instruction-effectiveness/analysis_util
export TRANSFORMERS_CACHE=/path/to/.cache/huggingface
port=$(shuf -i25000-30000 -n1)
echo $port
TASKS="task1670_md_gender_bias_text_modification"
for task in $TASKS
do
    deepspeed --master_port $port --include "localhost:2,3" src/run_input_reduce.py \
        --do_syntax_compress  \
        --predict_with_generate \
	--model_name_or_path /path/to/your/trained_model \
        --max_source_length 1024 \
        --max_target_length 128 \
        --generation_max_length 128 \
        --max_num_instances_per_task 100 \
        --max_num_instances_per_eval_task 100 \
	--test_on_specific_task $task \
        --add_task_name False \
        --add_task_definition True \
        --num_pos_examples 2 \
        --num_neg_examples 0 \
        --add_explanation False \
        --tk_instruct False \
	--parsed_file_dir analysis_util/parsed_instructions \
        --data_dir data/splits/default/holdout_separate \
        --task_dir data/tasks \
        --output_dir output/syntax_compress_dev \
        --overwrite_output_dir \
        --cache_dir ./cache/ \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-05 \
        --num_train_epochs 2 \
        --lr_scheduler_type constant \
        --warmup_steps 0 \
        --logging_strategy steps \
        --logging_steps 500 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --deepspeed ds_configs/stage3.config \
        --bf16 \
        --run_name t5-compression
done
