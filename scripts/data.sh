#!/bin/bash

language=$1
timestamp=$(date +%s)

LANGUAGE=${language} \
sbatch --output="./logs/processing/filter_${language}_${timestamp}.out" scripts/data.euler