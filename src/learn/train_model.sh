#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

DATE=$(date +%d%m%s)

DATA_DIR="data/${DATASET}-baseline"
MODEL_NAME="${MODEL}-${LANG}-${SEED}-${DATE}"

# Add extra flags for test training
if [ $DO_TEST = true ]
then
  extra_flags="--gradient_accumulation_steps 1 --logging_steps 1 --max_eval_samples 120 --max_train_samples 120"
fi

# Check if we resume training from checkpoint
if [ ! -z "${CHECKPOINT}" ]
then
  MODEL_NAME=${CHECKPOINT}
  extra_flags="--resume_from_checkpoint checkpoints/${CHECKPOINT}"
fi

# Select which model type to train
case $MODEL in
  gpt2)
    application="train_gpt2.sh"
    ;;

  *)
    echo -n "unknown model name"
    exit 1
    ;;
esac

# Train default tokenizer if it has not been trained before
if [ ! -z "${TOKENIZER}" ]
then
  echo "Using custom tokenizer ${TOKENIZER}"
  extra_flags="--tokenizer_name ${TOKENIZER}"
elif [ -d "data/counterfactual_${LANG}_tokenizer" ];
then
  echo "Default tokenizer is already trained."
elif [ -d "data/${LANG}_tokenizer" ];
then
  echo "Default tokenizer is already trained."
else
	echo "Did not find trained tokenizer. Training from scratch."
	python3 ./src/learn/train_tokenizer.py ${MODEL} ${DATASET} ${LANG}
fi

# Train on one language
export MODEL_NAME="${MODEL_NAME}"
export DATA_DIR=${DATA_DIR}

bash ./src/learn/${application} ${extra_flags}
