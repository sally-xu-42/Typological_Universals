#!/bin/bash

language="en"
do_test=False
timestamp=$(date +%s)

Help()
{
   # Display Help
   echo "Script to parse data into CoNLL-U format."
   echo
   echo "Syntax: parse.sh [-l|d|p|c|T]"
   echo "options:"
   echo "l     language code to parse, e.g. 'en,ja,zh-cn'. Default: en"
   echo "d     Path to data directory with plain txt files. Default: ./data/wiki40b-txt"
   echo "p     Path to store CONLLU parses. Default: ./parse"
   echo "c     Comma-separated list of partitions. Default: train,test,validation"
   echo "T     Test run."

   echo
}

while getopts "l:d:p:c:Th" option; do
  case $option in
    l)
      language="$OPTARG"
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
echo "Data directory: $data_dir"
echo "Parse directory: $parse_dir"
echo "List of partitions: $list_partitions"
echo "Test mode: $do_test"

# Set default value if ${TRAIN_SAMPLE} is not set or empty
# TRAIN_SAMPLE="${TRAIN_SAMPLE:-default_value}"

LANGUAGE=${language} DATA_DIR="${data_dir:-'./data/wiki40b-txt/'}" \
PARSE_DIR="${parse_dir:-'./parse'}" LIST_PARTITIONS="${list_partitions:-'train,test,validation'}" DO_TEST=${do_test} \
sbatch --output="./logs/parsing/dep_${language}_${timestamp}.out" scripts/depparse.euler