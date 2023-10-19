#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --output=slurm/slurm-%j.out
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g
{   
    export CUDA_VISIBLE_DEVICES=${11}
    train_val_dataset=$1
    test_dataset=$2
    epoch=$3
    seed=$4
    shot=$5
    context_scale=$6
    context_number=$7
    backbone=${8}
    lr=${9}
    module=${10}
    METHOD=CoOp_layerwise_partialft_${train_val_dataset}_${backbone}_${module}_lr_${lr}
    # METHOD=DATASET_${train_val_dataset}_backbone_${backbone}_CoOp_contextFT_lr_${lr}
    train_val_data_path=indices/${train_val_dataset}/shot_${shot}-seed_${seed}.json
    
    ###集群
    if test -d /tmp_space/CoOp; then
        test_data_path=/tmp_space/CoOp
    else
        test_data_path=data/CoOp
    fi

    ###本地
    # if test -d /space0/songk/datasets/CoOp/; then
    #     test_data_path=/space0/songk/datasets/CoOp
    # elif test -d /tmp_space/CoOp; then
    #     test_data_path=/tmp_space/CoOp
    # else
    #     test_data_path=/ceph/songk/datasets/CoOp
    # fi


    ###autodl
    # test_data_path=data/CoOp

    # results=ceph_result/zero_clip/${train_val_dataset}
    # results=ceph_result/head_backbone_${module}/${train_val_dataset}
    results=ceph_result/head_${backbone}_${module}/${train_val_dataset}

    mkdir -p ${results}


    TASK=CoOp
    # context_scale=50
    # context_number=20
    TASKID=${train_val_dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${lr}
    # TASKID=debug
    WANDB_PROJECT=CoOp_layerwise_partialft
    CONFIG_FILE=config.yaml
    CONFIG_PY=config/set_FT_CoOp_config_test.py
    ckpt_dir=${results}/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}

    # cd ${ckpt_dir}/checkpoints

    for file in ${ckpt_dir}/checkpoints/epoch*; do
        ckptPath=$file

    done
    echo $ckptPath

    mkdir -p ${ckpt_dir}/CoOp_${test_dataset}

    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${ckptPath} ${module} ${context_scale} ${context_number} ${ckpt_dir} CoOp_${train_val_dataset} ${train_val_data_path} CoOp_${test_dataset} ${test_data_path} ${lr} ${backbone} ${results} ${shot}
    # cp ${CONFIG_FILE} ${ckpt_dir}
    cp $0 ${ckpt_dir}/CoOp_${test_dataset}
    echo ${ckpt_dir}/CoOp_${test_dataset}/${CONFIG_FILE}
    python run.py --config ${ckpt_dir}/CoOp_${test_dataset}/${CONFIG_FILE}
    exit
}