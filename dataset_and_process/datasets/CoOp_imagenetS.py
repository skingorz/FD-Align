from torchvision import transforms
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from dataset_and_process.datasets.CoOp import CoOp
from tqdm import tqdm
from architectures.feature_extractor.clip import load
from PIL import Image
import json

class ImageNetS(ImageFolder):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="images", split_path="splits/split_zhou_Food101.json", image_sz = 84) -> None:
    # def __init__(self, data_root: str, mode: str, image_sz = 84) -> None:
        image_root = os.path.join(data_root, "imagenet-sketch/images")
        _, _, val_process=load(backbone_name, jit=False)
        self.transform = val_process
        super().__init__(image_root, self.transform)


def return_class():
    return ImageNetS

if __name__ == '__main__':
    # val = ImageNetA("indices/imagenet/shot_1-seed_1.json", "train", "RN50")
    test = ImageNetS("data/CoOp", "test")
    test.classes
    # val[1]
    test[1]

    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
