#!/bin/bash
{   
    export CUDA_VISIBLE_DEVICES=0
    # METHOD=debug
    METHOD=PN_FDAlign
    TASK=1shot
    TASKID=FDAlign
    # TASKID=debug
    WANDB_PROJECT=CLIPFT
    module=FD_Align_SCP_meta
    CONFIG_FILE=config/config.yaml
    CONFIG_PY=config/set_meta_config.py

    ckpt_dir=results/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}

    mkdir -p ${ckpt_dir}

    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${module}
    cp ${CONFIG_FILE} ${ckpt_dir}
    cp train.sh ${ckpt_dir}
    python run.py --config ${CONFIG_FILE}
    exit
}