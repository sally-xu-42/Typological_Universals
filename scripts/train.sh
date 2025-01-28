#!/bin/bash

model="gpt2"
lang="en"
dataset="wiki40b"
seed=42
project_name="typological-universals"
do_test=false
timestamp=$(date +%s)


Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-n|l|d|t|s|c|f|T|p]"
   echo "options:"
   echo "n     Model name (gpt2 or roberta). Default: gpt2"
   echo "l     Language to train on. Default: en"
   echo "d     Dataset txt files to use. Default: wiki40b"
   echo "t     Path to custom tokenizer"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training"
   echo "T     Run in test/debug mode (fewer samples). Default: false"
   echo "p     Project name for wandb logging. Default: typological-universals"

   echo
}

while getopts "n:l:d:t:s:c:p:Th" option; do
  case $option in
    n)
      model="$OPTARG"
      ;;
    l)
      lang="$OPTARG"
      ;;
    d)
      dataset="$OPTARG"
      ;;
    t)
      tokenizer="$OPTARG"
      ;;
    s)
      seed="$OPTARG"
      ;;
    c)
      checkpoint="$OPTARG"
      ;;
    p)
      project_name="$OPTARG"
      ;;
    T)
      do_test=true
      project_name="test"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-n model_name] [-d dataset] [-l language]"
      exit 1
      ;;
  esac
done

echo
echo "Model: $model"
echo "Language: $lang"
echo "Dataset: $dataset"
echo "Custom tokenizer: $tokenizer"
echo "Seed: $seed"
echo "Checkpoint: $checkpoint"
echo "Do test: $do_test"
echo "Project name: $project_name"

echo 'Doing normal training'
MODEL=${model} DATASET=${dataset} TOKENIZER=${tokenizer} LANG=${lang} SEED=${seed} \
PROJECT=${project_name} CHECKPOINT=${checkpoint} DO_TEST=${do_test} \
sbatch  --job-name="lm-train-${model}-${lang}" \
      --output="./logs/trainings/train_${model}_${lang}_${seed}_${timestamp}.out" \
      scripts/train.euler

echo