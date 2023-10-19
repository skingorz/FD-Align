#!/bin/bash
#SBATCH --partition=3090
#SBATCH --exclude=3dimage-20
#SBATCH --output=slurm/slurm-%j.out
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g

dataset=$1
epoch=$2
seed=$3
shot=$4
context_scale=$5
context_number=$6
backbone=$7
lr_config=$8
module=$9
gpuid=${10}
optim=${11}
load_head=${12}
if test -d ceph_result/head_${backbone}_${module}/${dataset}/CoOp_layerwise_partialft_${dataset}_${backbone}_${module}_lr_${lr_config}/CoOp/CoOp_layerwise_partialft/${dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${lr_config}/checkpoints; then
    echo "have trained"
else
    bash train_CoOp.sh $dataset $epoch $seed $shot $context_scale $context_number $backbone $lr_config $module $gpuid $optim $load_head
fi


if test -f ceph_result/head_${backbone}_${module}/${dataset}/CoOp_layerwise_partialft_${dataset}_${backbone}_${module}_lr_${lr_config}/CoOp/CoOp_layerwise_partialft/${dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${lr_config}/CoOp_${dataset}/results/test_result.json; then
    echo "have tested"
else
    bash test_CoOp.sh $dataset $dataset $epoch $seed $shot $context_scale $context_number $backbone $lr_config $module $gpuid
fi
