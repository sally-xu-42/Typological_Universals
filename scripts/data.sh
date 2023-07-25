#!/bin/bash

language="en"
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run data processing."
   echo
   echo "Syntax: data.sh [-l|t|s|v|o|b|d]"
   echo "options:"
   echo "l     Comma-separated list of language codes to use, e.g. 'en,de,nl,hu'. Default: en"
   echo "t     Max number of training examples to sample. Default: -1.0"
   echo "s     Max number of test examples to sample. Default: -1.0"
   echo "v     Max number of validation examples to sample. Default: -1.0"
   echo "o     Path to output destination for dataset text files. Default: ./data/wiki40b-txt/"
   echo "b     Batch size for downloading. Default: 128"
   echo "d     Path to save data files to. Default: ./data/"

   echo
}

while getopts "l:t:s:v:o:b:dh" option; do
  case $option in
    l)
      language="$OPTARG"
      ;;
    t)
      train_sample="$OPTARG"
      ;;
    s)
      test_sample="$OPTARG"
      ;;
    v)
      valid_sample="$OPTARG"
      ;;
    o)
      output_dir="$OPTARG"
      ;;
    b)
      batch_size="$OPTARG"
      ;;
    d)
      save_dir="$OPTARG"
      ;;
    h)
      Help
      exit
      ;;
    *)
      echo "Usage: $0 [-l language]"
      exit 1
      ;;
  esac
done

echo
echo "Language list: $language"
echo "Training samples: $train_sample"
echo "Testing samples: $test_sample"
echo "Validation samples: $valid_sample"
echo "Path for output text: $output_dir"
echo "Batch size: $batch_size"
echo "Path for saving dataset: $save_dir"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

LANGUAGE=${language} TRAIN_SAMPLE="${train_sample:--1.0}" TEST_SAMPLE="${test_sample:--1.0}" \
VALID_SAMPLE="${valid_sample:--1.0}" OUTPUT_DIR="${output_dir:-'./data/wiki40b-txt/'}" BATCH_SIZE="${batch_size:-128}" SAVE_DIR="${save_dir:-'./data/'}" \
sbatch --output="./logs/processing/download_${language}_${timestamp}.out" scripts/data.euler