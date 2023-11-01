#!/bin/bash

{
    datasets=(aircraft caltech101 dtd eurosat imagenet flowers102 food101 oxfordpets StandfordCars sun397 ucf101)
    gpuid=0
    seed=1
    shots=(1 2 4 8 16)
    epoch=60
    context_scale=20
    context_numbers=(20)
    backbones=(ViT_B_32_clip)
    load_head=False
    modules=(LP_CoOp)
    optim=sgd

    for context_number in "${context_numbers[@]}"
    do
        for module in "${modules[@]}"
        do
            for backbone in "${backbones[@]}"
            do
                for dataset in "${datasets[@]}"
                do
                    for shot in "${shots[@]}"
                    do  
                        lr_config=${dataset}_config
                        if test -f ceph_result/head_${backbone}_${module}/${dataset}/CoOp_layerwise_partialft_${dataset}_${backbone}_${module}_lr_${lr_config}/CoOp/CoOp_layerwise_partialft/${dataset}_epoch=${epoch}_seed_${seed}_shot_${shot}_${context_scale}_${module}_${context_number}_lr_${lr_config}/CoOp_${dataset}/results/test_result.json; then
                            echo have tested
                        else
                            bash train_test.sh $dataset $epoch $seed $shot $context_scale $context_number $backbone $lr_config $module $gpuid $optim $load_head
                        fi
                    done
                done
            done
        done
    done
}
