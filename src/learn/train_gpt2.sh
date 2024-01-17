#!/bin/bash

export NCCL_DEBUG=INFO

EXTRA_FLAGS="$@"
echo $EXTRA_FLAGS

RUN_APPLICATION="torchrun --nproc_per_node 8"

${RUN_APPLICATION} ./src/learn/run_clm.py \
    --model_type gpt2 \
    --tokenizer_name "data/tokenizer/${DATASET}-${LANG}/gpt2_tokenizer" \
    --train_file "${DATA_DIR}/${LANG}_train.txt" \
    --validation_file "${DATA_DIR}/${LANG}_validation.txt" \
    --cache_dir "data/cache" \
    --run_name ${MODEL_NAME} \
    --seed ${SEED} \
    --report_to wandb \
    --output_dir "./checkpoints/${MODEL_NAME}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --eval_accumulation_steps 128 \
    --block_size 512 \
    --do_train \
    --logging_steps 50 \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-3 \
    --warmup_ratio 0.07 \
    --num_train_epochs 50 \
    --low_cpu_mem_usage \
    --fp16 \
    --ddp_backend nccl \
    --ddp_timeout 18000 \
    ${EXTRA_FLAGS}