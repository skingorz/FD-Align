#!/bin/bash
#SBATCH --partition=3090
#SBATCH --output=slurm/slurm-%j.out
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g
{
    export CUDA_VISIBLE_DEVICES=${10}
    train_val_dataset=$1
    test_dataset=$1
    epoch=$2
    shot=$4
    seed=$3
    backbone=${7}
    lr=${8}
    module=${9}
    optim=${11}
    load_head=${12}
    METHOD=CoOp_layerwise_partialft_${train_val_dataset}_${backbone}_${module}_lr_${lr}
    train_val_data_path=indices/${train_val_dataset}/shot_${shot}-seed_${seed}.json

    ## 集群
    if test -d /tmp_space/CoOp; then
        test_data_path=/tmp_space/CoOp
    else
        test_data_path=data/CoOp
    fi

    echo ${test_data_path}

    ##本地
    # if test -d /space0/songk/datasets/CoOp/; then
    #     test_data_path=/space0/songk/datasets/CoOp
    # elif test -d /tmp_space/CoOp; then
    #     test_data_path=/tmp_space/CoOp
    # else
    #     test_data_path=/ceph/songk/datasets/CoOp
    # fi


    ### autodl
    # test_data_path=data/CoOp

    TASK=CoOp
    context_scale=$5
    context_number=$6
    TASKID=${train_val_dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${lr}
    # TASKID=debug
    WANDB_PROJECT=CoOp_layerwise_partialft
    CONFIG_FILE=config.yaml
    CONFIG_PY=config/set_FT_CoOp_config.py
    results=ceph_result/head_${backbone}_${module}/${train_val_dataset}
    # results=ceph_result/head_full_${module}/${train_val_dataset}

    # if [$head_lr]; then
    #     head_lr=${head_lr}
    #     head_ckpt=head_result/${train_val_dataset}/CoOp_layerwise_partialft_${train_val_dataset}_${backbone}_${module}_lr_${head_lr}/CoOp/CoOp_layerwise_partialft/${train_val_dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${head_lr}/checkpoints/last.ckpt
    # else
    #     head_lr=None
    #     head_ckpt=None
    # fi

    mkdir -p ${results}

    # 集群
    if test -d /tmp; then
        log_dir=/tmp/
        ckpt_dir=/tmp/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}
    else
        log_dir=./${results}/
        ckpt_dir=${results}/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}
    fi

    # autodl
    # log_dir=./${results}/
    # ckpt_dir=${results}/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}

    mkdir -p ${ckpt_dir}

    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${module} ${log_dir} ${context_scale} ${context_number} ${ckpt_dir} CoOp_${train_val_dataset} ${train_val_data_path} CoOp_${test_dataset} ${test_data_path} ${lr} ${backbone} ${epoch} ${optim} ${load_head} ${shot} ${seed}

    cp $0 ${ckpt_dir}
    echo ${ckpt_dir}/${CONFIG_FILE}
    python run.py --config ${ckpt_dir}/${CONFIG_FILE}
    if test -d /tmp; then
        mkdir -p ${results}/${METHOD}/${TASK}/${WANDB_PROJECT}
        rsync -a /tmp/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID} ${results}/${METHOD}/${TASK}/${WANDB_PROJECT}
        mkdir -p ${results}/${METHOD}/${TASK}/wandb
        rsync -a /tmp/${METHOD}/${TASK}/wandb/*${TASKID} ${results}/${METHOD}/${TASK}/wandb
        rm -rf /tmp/${METHOD}/${TASK}/wandb/*${TASKID}
        rm -rf /tmp/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}
    fi
    exit
}