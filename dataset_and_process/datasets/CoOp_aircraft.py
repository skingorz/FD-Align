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

class aircraft(CoOp):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="images", split_path="images_variant_test.txt", image_sz = 84) -> None:
        self.image_root = os.path.join(data_root, "fgvc_aircraft", image_root)
        assert mode in ["train", "val", "test"]
        self.mode = mode
        if mode == "train" or mode == "val":
            super().__init__(data_root, mode, backbone_name, self.image_root, split_path, image_sz)
        else:
            _, _, val_process = load(backbone_name, jit=False)
            self.transform = val_process
            assert mode == "test"
            classname = []
            with open(os.path.join(data_root, "fgvc_aircraft", "variants.txt"), "r") as f:
                for line in f:
                    classname.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classname)}

            self.split_path = os.path.join(data_root, "fgvc_aircraft", split_path)
            self.image_path = []
            self.label = []
            with open(self.split_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    imname = line[0] + ".jpg"
                    classname = " ".join(line[1:])
                    impath = os.path.join(self.image_root, imname)
                    label = cname2lab[classname]

                    self.image_path.append(impath)
                    self.label.append(label)



def return_class():
    return aircraft

if __name__ == '__main__':
    # train = caltech101("indices/fgvc_aircraft/shot_1-seed_1.json", "train")
    val = aircraft("indices/aircraft/shot_1-seed_1.json", "val", "RN50")
    test = aircraft("data/CoOp", "test", "RN50")
    val[1]
    test[1]
    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
