# FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning

Officaial implementation of FD-Align for paper "FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning" (NeurIPS 2023)

This code is based on [LightningSL](https://github.com/Frankluox/LightningFSL). You can refer to [corss modal adaptation](https://github.com/linzhiqiu/cross_modal_adaptation) and [Channel_Importance_FSL](https://github.com/Frankluox/Channel_Importance_FSL) to prepare the data.




## CoOp task

1. Fine-tune the classification head (Optional).

```
bash tools/LP_alldataset.sh
```

2. FD-Align Fine-tune

```
bash tools/FT_alldataset.sh
```

## N-way-K-shot task

### Train

Fine tune model on miniImageNet.

```
bash tools/train_meta.sh
```

### Test

Evaluate performance on different datasets.

```
bash tools/test_meta.sh
```

## Citation

If our code is helpful for your research, please cite the following paper:

```
@article{song2023FD,
    title={FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning},
    author={Kun Song and Huimin Ma and Bochao Zou and Huishuai Zhang and Weiran Huang},
    journal={NeurIPS},
    year={2023}
}
```
