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

class ImageNetV2(ImageFolder):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="images", split_path="splits/split_zhou_Food101.json", image_sz = 84) -> None:
    # def __init__(self, data_root: str, mode: str, image_sz = 84) -> None:
        image_root = os.path.join(data_root, "imagenetv2/imagenetv2-matched-frequency-format-val")
        self.image_path = []
        self.label = []
        for i in range(1000):
            image_fold = os.path.join(image_root, str(i))
            for image_name in os.listdir(image_fold):
                self.image_path.append(os.path.join(image_fold, image_name))
                self.label.append(i)
        _, _, val_process=load(backbone_name, jit=False)

        self.transform = val_process


    def __getitem__(self, index: int):
        image = Image.open(self.image_path[index]).convert('RGB')
        image = self.transform(image)

        return image, self.label[index]

    def __len__(self):
        return len(self.image_path)

def return_class():
    return ImageNetV2

if __name__ == '__main__':
    # val = ImageNetA("indices/imagenet/shot_1-seed_1.json", "train", "RN50")
    test = ImageNetV2("data/CoOp", "test")
    test.classes
    # val[1]
    test[1]

    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
