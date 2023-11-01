#!/bin/bash
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --partition=3090
#SBATCH -n 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 50g
{   
    export CUDA_VISIBLE_DEVICES=2
    
    # DATA_ROOT=/tmp_space/meta-dataset
    DATA_ROOT=data/meta-dataset

    # DATASET=miniImageNet_clip
    # DATAPATH=${DATA_ROOT}/mini_imagenet/mini_imagenet_split
    
    # DATASET=cub_clip
    # DATAPATH=${DATA_ROOT}/cub

    # DATASET=clipart_clip
    # DATAPATH=${DATA_ROOT}/clipart

    # DATASET=euro_clip
    # DATAPATH=${DATA_ROOT}/EuroSAT

    # DATASET=chestX_clip
    # DATAPATH=${DATA_ROOT}/ChestX/data

    # DATASET=coco_clip
    # DATAPATH=${DATA_ROOT}/coco

    # DATASET=Textures_clip
    # DATAPATH=${DATA_ROOT}/Textures/dtd/images

    # DATASET=aircraft_clip
    # DATAPATH=${DATA_ROOT}/fgvc-aircraft-2013b/data

    # DATASET=trafficsign_clip
    # DATAPATH=${DATA_ROOT}/GTSRB/Final_Training/Images

    # check!!!!!!!!!!!!!!!
    DATASET=omniglot_clip
    DATAPATH=${DATA_ROOT}/omniglot

    # DATASET=vggflower_clip
    # DATAPATH=${DATA_ROOT}/VGGFlower

    # DATASET=quicklydraw_clip
    # DATAPATH=${DATA_ROOT}/quickdraw

    # DATASET=fungi_clip
    # DATAPATH=${DATA_ROOT}/fungi_train_val/images

    # DATASET=plantDIsease_clip
    # DATAPATH=${DATA_ROOT}/PlantDisease/train

    # DATASET=ISIC_clip
    # DATAPATH=${DATA_ROOT}/ISIC

    # DATASET=real_clip
    # DATAPATH=${DATA_ROOT}/real

    # DATASET=sketch_clip
    # DATAPATH=${DATA_ROOT}/sketch

    # DATASET=infograph_clip
    # DATAPATH=${DATA_ROOT}/infograph

    # DATASET=painting_clip
    # DATAPATH=${DATA_ROOT}/painting

    METHOD=FDAlign
    TASK=1shot
    TASKID=FDAlign
    WANDB_PROJECT=CLIPFT
    model=epoch=4-step=1249.ckpt      # ckpt file name
    # CONFIG_DIR=config
    CONFIG_FILE=config_test.yaml
    CONFIG_PY=config/set_meta_config_test.py # python file to set config

    TASK_Dir=results/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}/${DATASET}

    mkdir -p ${TASK_Dir}

    ckpt_dir=results/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}
    ckptPath=${ckpt_dir}/checkpoints/${model}

    # 1shot config
    mkdir ${TASK_Dir}/1shot_results
    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${ckptPath} ${DATASET} ${DATAPATH} ${ckpt_dir} 1
    cp ${ckpt_dir}/${CONFIG_FILE} ${TASK_Dir}/1shot_results
    cp $0 ${TASK_Dir}/1shot_results

    # 5shot config
    mkdir ${TASK_Dir}/5shot_results
    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${ckptPath} ${DATASET} ${DATAPATH} ${ckpt_dir} 5
    cp ${ckpt_dir}/${CONFIG_FILE} ${TASK_Dir}/5shot_results
    cp $0 ${TASK_Dir}/5shot_results

    # test 1shot
    python run.py --config ${TASK_Dir}/1shot_results/${CONFIG_FILE}

    # test 5shot
    python run.py --config ${TASK_Dir}/5shot_results/${CONFIG_FILE}

    exit
}