#!/bin/bash

module purge
module load eth_proxy gcc/8.2.0 python_gpu/3.9.9
source /cluster/work/sachan/txu/Typological_Universal/env2/bin/activate


config_name="gpt2"
project_name="typological-universals"


while getopts "n:l:p:" option; do
case $option in
n)
config_name="$OPTARG"
;;
l)
lang="$OPTARG"
;;
p)
project_name="$OPTARG"
;;
*)
echo "Usage: $0 [-n config_name] [-l language] [-p project_name]"
exit 1
;;
esac
done


wandb sweep --project ${project_name} --name "${config_name}-${lang}-sweep" ./src/learn/configs/sweep_${config_name}.yaml