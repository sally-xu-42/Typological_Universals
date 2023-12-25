#!/bin/bash

language="en"
timestamp=$(date +%s)

Help()
{
    # Display Help
    echo "Script to sample 100 random sentences from training split's parse."
    echo
    echo "Syntax: sample.sh [-l|p|s]"
    echo "options:"
    echo "l     language code to parse, e.g. 'en,ja,zh-cn'. Default: en"
    echo "p     Path to CONLLU parse directory. Default: ./parse"
    echo "s     Path to store 100 sample sentences,both original text and their parses. Default: ./data/wiki40b-manual"

    echo
}

while getopts "l:p:s:h" option; do
  case $option in
    l)
      language="$OPTARG"
      ;;
    p)
      parse_dir="$OPTARG"
      ;;
    s)
      sample_dir="$OPTARG"
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
echo "Parse directory: $parse_dir"
echo "Sample directory: $sample_dir"

LANGUAGE=${language} PARSE_DIR="${parse_dir:-'./parse'}" SAMPLE_DIR="${sample_dir:-'./data/wiki40b-manual'}" \
sbatch --output="./logs/sampling/sample_${language}_${timestamp}.out" scripts/sample.euler

