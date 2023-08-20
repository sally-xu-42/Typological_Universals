#!/bin/bash

language="en"
do_test=False
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to parse data into CoNLL-U format."
   echo
   echo "Syntax: parse.sh [-l|u|d|p|c|T]"
   echo "options:"
   echo "l     2-letter language code to parse, e.g. 'en,ja'. Default: en"
   echo "u     Path to UDPipe model file for this language. Default: udpipe_models/english-lines-ud-2.5-191206.udpipe"
   echo "d     Path to data directory with plain txt files. Default: ./data/wiki40b-txt"
   echo "p     Path to store CONLLU parses. Default: ./parse"
   echo "c     Comma-separated list of partitions. Default: train,test,validation"
   echo "T     Test run."

   echo
}

while getopts "l:u:d:p:c:Th" option; do
  case $option in
    l)
      language="$OPTARG"
      ;;
    u)
      model_path="$OPTARG"
      ;;
    d)
      data_dir="$OPTARG"
      ;;
    p)
      parse_dir="$OPTARG"
      ;;
    c)
      list_partitions="$OPTARG"
      ;;
    T)
      do_test=true
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
echo "UDPipe model path: $model_path"
echo "Data directory: $data_dir"
echo "Parse directory: $parse_dir"
echo "List of partitions: $list_partitions"
echo "Test mode: $do_test"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

LANGUAGE=${language} MODEL_PATH="${model_path:-'udpipe_models/english-lines-ud-2.5-191206.udpipe'}" DATA_DIR="${data_dir:-'./data/wiki40b-txt/'}" \
PARSE_DIR="${parse_dir:-'./parse'}" LIST_PARTITIONS="${list_partitions:-'train,test,validation'}" DO_TEST=${do_test} \
sbatch --output="./logs/parsing/dep_${language}_${timestamp}.out" scripts/depparse.euler