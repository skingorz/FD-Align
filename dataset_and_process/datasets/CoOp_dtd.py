from torchvision import transforms
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from dataset_and_process.datasets.CoOp import CoOp
from architectures.feature_extractor.clip import load
from PIL import Image
import json

class DTD(CoOp):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="images", split_path="splits/split_zhou_DescribableTextures.json", image_sz = 84) -> None:
        self.image_root = os.path.join(data_root, "dtd", image_root)
        super().__init__(data_root, mode, backbone_name, self.image_root, split_path, image_sz)


def return_class():
    return DTD

if __name__ == '__main__':
    # train = caltech101("indices/dtd/shot_1-seed_1.json", "train")
    val = DTD("indices/dtd/shot_1-seed_1.json", "val")
    test = DTD("data/CoOp/dtd", "test")
    val[1]
    test[1]
    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
