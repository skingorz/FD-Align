from torchvision import transforms
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import pickle
from architectures.feature_extractor.clip import load

class miniImageNet(ImageFolder):
    r"""The standard  dataset for miniImageNet. ::
    
        root
        |
        |
        |---train
        |    |--n01532829
        |    |   |--n0153282900000005.jpg
        |    |   |--n0153282900000006.jpg
        |    |              .
        |    |              .
        |    |--n01558993
        |        .
        |        .
        |---val
        |---test  
    Args:
        root: Root directory path.
        mode: train or val or test
    """
    def __init__(self, root: str, mode: str, backbone_name="resnet12", image_sz = 84) -> None:
        assert mode in ["train", "val", "test"]
        self.mode = mode
        _, train_process, val_process=load(backbone_name, jit=False)
        # mode = "test" if mode == "val" else "val"
        IMAGE_PATH = os.path.join(root, mode)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process

        super().__init__(IMAGE_PATH, transform)
        self.label = self.targets
        

def return_class():
    return miniImageNet

if __name__ == '__main__':
    val = miniImageNet("data/mini_imagenet/mini_imagenet_split", "val")
    train = miniImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    test = miniImageNet("data/mini_imagenet/mini_imagenet_split", "test")
