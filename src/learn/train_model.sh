#!/bin/bash

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=$PROJECT

DATE=$(date +%d%m)

DATA_DIR="data/${DATASET}-txt/en.small"       ## testing on small dataset, always run from root directory. Obtain processed data before you run this shell script.
MODEL_NAME="${MODEL}-${CONFIG}-${LANG}-${MODE}-${SEED}-${DATE}"

extra_flags=$(<"./src/learn/configs/${MODEL}_${CONFIG}.txt")

# Add extra flags for test training
if [ $DO_TEST = true ]
then
  extra_flags="${extra_flags} --gradient_accumulation_steps 1 --logging_steps 1 --max_eval_samples 120 --max_train_samples 120"
fi

# Check if we resume training from checkpoint, removed second language check
if [ ! -z "${CHECKPOINT}" ]
then
  MODEL_NAME=${CHECKPOINT}
  extra_flags="${extra_flags} --resume_from_checkpoint checkpoints/${CHECKPOINT}"
fi

# Select which model type to train
case $MODEL in
  gpt2)
    application="train_gpt2.sh"
    ;;

  roberta)
    application="train_roberta.sh"
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
  extra_flags="${extra_flags} --tokenizer_name data/${DATASET}/${TOKENIZER}"

elif [ -d "data/tokenizer/${DATASET}-${LANG}/${MODEL}_tokenizer" ];
then
  echo "Default tokenizer is already trained."
else
	echo "Did not find trained tokenizer. Training from scratch."
	python3 ./src/learn/train_tokenizer.py ${MODEL} ${DATASET} ${LANG}
fi


if [ ! -z "${SWEEP_ID}" ]
then
  # Hyperparameter sweep
  export MODEL_NAME="${MODEL}-${LANG}-${SEED}-${DATE}-${IDX}"
  export DATA_DIR=${DATA_DIR}

  wandb agent --count 5 ${SWEEP_ID}

else
    # Train on one language
    export MODEL_NAME="${MODEL_NAME}"
    export DATA_DIR=${DATA_DIR}

    bash ./src/learn/${application} ${extra_flags}

fi