#!/bin/bash

timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run evaluation."
   echo
   echo "Syntax: evaluate.sh [-t|m|e]"
   echo "options:"
   echo "t     Evaluation task. Default: blimp"
   echo "m     Model checkpoint directory for evaluation. Default: gpt2--en-42-2211"
   echo "e     Encoder or decoder for BLiMP. Default: decoder"

   echo
}

while getopts "t:m:e:h" option; do
  case $option in
    t)
      task="$OPTARG"
      ;;
    m)
      model="$OPTARG"
      ;;
    e)
      model_type="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-m gpt2--en-42-2211]"
      exit 1
      ;;
  esac
done

echo
echo "Evaluation task: $task"
echo "Model checkpoint directory for evaluation: $model"
echo "Encoder/Decoder model type: $model_type"

TASK=${task} MODEL=${model} MODEL_TYPE=${model_type} \
sbatch  --job-name="evaluate-${task}-${model}" \
        --output="./logs/evaluations/evaluate_${task}_${model}_${timestamp}.out" \
        scripts/evaluate.euler