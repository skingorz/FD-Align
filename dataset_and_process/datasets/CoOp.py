from torchvision import transforms
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from architectures.feature_extractor.clip import load
from PIL import Image
import json

class CoOp(ImageFolder):

    # if mode == train or val, data root is the path to the folder that contains the indices
    # if mode == test, data root is the path to the folder that contains the images
    def __init__(self, data_root: str, mode: str, backbone_name="resnet12", image_root="", split_path="split", image_sz = 84) -> None:
        assert mode in ["train", "val", "test"]
        _, train_process, val_process=load(backbone_name, jit=False)
        self.mode = mode
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        self.transform = transform

        # self.split_path(data_root)

        if self.mode == "train" or self.mode == "val":
            with open(data_root, 'r') as f:
                data = json.load(f)
            data = data[mode]['data']
            self.image_path = []
            self.label = []
            for i in tqdm(range(len(data))):
                if os.path.exists("/tmp_space/CoOp"):
                    self.image_path.append(data[i]['impath'].replace("data/CoOp","/tmp_space/CoOp"))
                else:
                    self.image_path.append(data[i]['impath'])
                self.label.append(int(data[i]['label']))
        
        else:
            self.image_root = image_root
            self.split_path = split_path
            with open(self.split_path, 'r') as f:
                data = json.load(f)
            test_data = data['test']
            self.image_path = []
            self.label = []
            self.classname = []
            clab2name = {}
            for impath, label, classname in test_data:
                self.image_path.append(os.path.join(self.image_root, impath))
                self.label.append(int(label))
                self.classname.append(classname)
                clab2name[int(label)] = classname
            

                
                # test_data.append({'impath': impath, 'label': int(label), 'classname': classname})
            


    def __getitem__(self, index: int):
        image = Image.open(self.image_path[index]).convert('RGB')
        image = self.transform(image)

        return image, self.label[index]

    def __len__(self):
        return len(self.image_path)

def return_class():
    return CoOp

if __name__ == '__main__':
    # train = caltech101("indices/ucf101/shot_1-seed_1.json", "train")
    # val = caltech101("indices/ucf101/shot_1-seed_1.json", "val")
    test = CoOp("data/CoOp/ucf101", "test")
    test[1]
    # train = ImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    # test = ImageNet("data/mini_imagenet/mini_imagenet_split", "test")
