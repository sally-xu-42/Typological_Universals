#!/bin/bash
export STANZA_RESOURCES_DIR="$SLURM_SUBMIT_DIR"

language="en"
do_test=False
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to parse data into CoNLL-U format."
   echo
   echo "Syntax: parse.sh [-l|d|b]"
   echo "options:"
   echo "l     language code to remove punctuation, e.g. 'en,ja,zh-cn'. Default: en"
   echo "d     Path to data directory with plain txt files. Default: ./data/wiki40b-txt"
   echo "b     Path to output directory with txt files with punctuations removed. Default: ./data/wiki40b-baseline"

   echo
}

while getopts "l:d:bh" option; do
  case $option in
    l)
      language="$OPTARG"
      ;;
    d)
      data_dir="$OPTARG"
      ;;
    b)
      output_dir="$OPTARG"
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
echo "Language code: $language"
echo "Data directory: $data_dir"
echo "Output directory: $output_dir"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

LANGUAGE=${language} DATA_DIR="${data_dir:-'./data/wiki40b-txt'}" \
OUTPUT_DIR="${parse_dir:-'./data/wiki40b-baseline'}" \
sbatch --output="./logs/baseline/remove_${language}_${timestamp}.out" scripts/remove_punct.euler