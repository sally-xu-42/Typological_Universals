#!/bin/bash

language="en"
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run counterfactual corpus creation."
   echo
   echo "Syntax: counterfactual.sh [-l|p|f|o]"
   echo "options:"
   echo "l     Language code to create counterfactual corpus. Default: en"
   echo "p     Pair <X,Y> to swap. e.g.: VO, ADP_NP. Default: VO"
   echo "f     Filename of CONLLU data. Default: ./data/wiki40b-random/regular/en_sample.conllu"
   echo "o     Output file of plain reordered text that has a specific <X,Y> pair reordered. Default: ./data/wiki40b-random/regular/en_ov_test.txt"

   echo
}

while getopts "l:p:f:o:h" option; do
  case $option in
    l)
      language="$OPTARG"
      ;;
    p)
      swap_pair="$OPTARG"
      ;;
    f)
      conllu_file="$OPTARG"
      ;;
    o)
      output_file="$OPTARG"
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
echo "Pair to reverse: $swap_pair"
echo "CoNLLU parse file: $conllu_file"
echo "Counterfactual corpus stored at: $output_file"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

LANGUAGE=${language} SWAP_PAIR="${swap_pair:-'VO'}" \
CONLLU_FILE="${conllu_file:-'./parse/en_train.conllu'}" \
OUTPUT_FILE="${output_file:-'./data/wiki40b-OV/en/en_ov_train.txt'}" \
sbatch --output="./logs/counterfactual/rev_${language}_${swap_pair}_${timestamp}.out" scripts/counterfactual.euler