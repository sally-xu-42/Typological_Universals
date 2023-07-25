#!/bin/bash

model="gpt2"
lang="en"
dataset="wiki40b-txt"                               ## dataset="unified_clean"
seed=0
project_name="typological-universals"
do_sweep=false
do_test=false
timestamp=$(date +%s)
# training_mode="sequential"

Help()
{
   # Display Help
   echo "Script to run model training."
   echo
   echo "Syntax: train.sh [-n|l|d|t|s|c|f|T|p|w]"
   echo "options:"
   echo "n     Model name (gpt2 or roberta). Default: gpt2"
   echo "l     Language to train on. Default: en"
   echo "d     Dataset txt files to use. Default: wiki40b-txt"
   echo "t     Path to custom tokenizer"
   echo "s     Random seed number. Default: 42"
   echo "c     Checkpoint path to resume training"
   echo "f     Config file for model (extra flags)"
   echo "T     Run in test/debug mode (fewer samples). Default: false"
   echo "p     Project name for wandb logging. Default: typological-universals"
   echo "w     Wandb sweep id for hyperparameter tuning (sweep must be started already)"

   echo
}

while getopts "n:l:d:t:s:c:f:p:w:Th" option; do
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
    f)
      config_file="$OPTARG"
      ;;
    p)
      project_name="$OPTARG"
      ;;
    w)
      sweep_id="$OPTARG"
      do_sweep=true
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
echo "Configuration file: $config_file"
echo "Do test: $do_test"
echo "Do sweep: $do_sweep"
echo "Project name: $project_name"

if [ $do_sweep = true ] ;
then
  echo 'Doing hyperparameter sweep'

  # Run sweep agents in parallel
  for i in 1 2 3 4
  do
    MODEL=${model} DATASET=${dataset} TOKENIZER=${tokenizer} LANG=${lang} SEED=${seed} \
    PROJECT=${project_name} DO_TEST=${do_test} SWEEP_ID=${sweep_id} IDX=${i} \
    sbatch  --job-name="sweep-${model}-${lang}" \
            --output="./logs/sweeps/sweep_${model}_${lang}_${seed}_${timestamp}_${i}.out" \
            scripts/train.euler
  done

else
  echo 'Doing normal training'
  MODEL=${model} DATASET=${dataset} TOKENIZER=${tokenizer} LANG=${lang} SEED=${seed} \
  PROJECT=${project_name} CHECKPOINT=${checkpoint} CONFIG=${config_file} DO_TEST=${do_test} \
  sbatch  --job-name="lm-train-${model}-${lang}" \
          --output="./logs/trainings/train_${model}_${config_file}_${lang}_${seed}_${timestamp}.out" \
          scripts/train.euler
fi


echo