#!/bin/bash

language=$1
timestamp=$(date +%s)

LANGUAGE=${language} \
sbatch --output="./logs/processing/download_${language}_${timestamp}.out" scripts/data.euler