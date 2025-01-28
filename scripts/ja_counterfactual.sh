#!/bin/bash

timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to run counterfactual corpus creation."
   echo
   echo "Syntax: counterfactual.sh [-p|f|o]"
   echo "options:"
   echo "p     Pair <X,Y> to swap. e.g.: VO, ADP_NP. Default: VO"
   echo "f     Filename of CONLLU data. Default: ./data/wiki40b-random/regular/en_sample.conllu"
   echo "o     Output file of plain reordered text that has a specific <X,Y> pair reordered. Default: ./data/wiki40b-random/regular/en_ov_test.txt"

   echo
}

while getopts "p:f:o:h" option; do
  case $option in
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

swap_pair="vo"
conllu_file="./parse/ja_test.conllu"
output_file="./sample/ja_${swap_pair}_sample.txt"

echo
echo "Pair to reverse: $swap_pair"
echo "CoNLLU parse file: $conllu_file"
echo "Counterfactual corpus stored at: $output_file"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

SWAP_PAIR=${swap_pair} \
CONLLU_FILE=${conllu_file} \
OUTPUT_FILE=${output_file} \
sbatch --output="./logs/counterfactual/rev_${swap_pair}_${timestamp}.out" scripts/ja_counterfactual.euler
