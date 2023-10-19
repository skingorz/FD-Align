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

class StandfordCars(CoOp):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="", split_path="splits/split_zhou_StanfordCars.json", image_sz = 84) -> None:
        self.image_root = os.path.join(data_root, "stanford_cars", image_root)
        super().__init__(data_root, mode, backbone_name, self.image_root, split_path, image_sz)

    def __getitem__(self, index: int):
        image = Image.open(self.image_path[index]).convert('RGB')
        image = self.transform(image)

        return image, self.label[index]

    def __len__(self):
        return len(self.image_path)

def return_class():
    return StandfordCars

if __name__ == '__main__':
    # train = StandfordCars("indices/stanford_cars/shot_1-seed_1.json", "train")
    # val = StandfordCars("indices/stanford_cars/shot_1-seed_1.json", "val")
    test = StandfordCars("data/CoOp", "test")
    test[1]
    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
