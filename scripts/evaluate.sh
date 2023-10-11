#!/bin/bash

task=$1
model=$2
model_type=$3
timestamp=$(date +%s)

TASK=${task} MODEL=${model} MODEL_TYPE=${model_type} \
sbatch  --job-name="evaluate-${task}-${model}" \
        --output="./logs/evaluations/evaluate_${task}_${model}_${timestamp}.out" \
        scripts/evaluate.euler