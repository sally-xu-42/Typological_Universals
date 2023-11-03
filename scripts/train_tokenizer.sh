#!/bin/bash

model=$1
dataset=$2
language=$3
timestamp=$(date +%s)

MODEL=${model} DATASET=${dataset} LANGUAGE=${language} \
sbatch  --job-name="train-tokenizer-${dataset}-${language}" \
        --output="./logs/trainings/train_tokenizer_${dataset}_${language}_${timestamp}.out" \
        scripts/train_tokenizer.euler