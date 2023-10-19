from torchvision import transforms
import os
import torch
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder
from architectures.feature_extractor.clip import load, tokenize, get_zeroshot_weight
from .openai_imagenet_temple import openai_imagenet_template
from .class_name import mini_train, mini_val, mini_test

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
        clip_model, train_process, val_process=load(backbone_name, jit=False)
        # mode = "test" if mode == "val" else "val"
        IMAGE_PATH = os.path.join(root, mode)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        if mode == "train":
            classnames = mini_train
        elif mode == "val":
            classnames = mini_val
        elif mode == "test":
            classnames = mini_test
            
        self.mode = mode
        self.class_embedding = []
        for params in list(clip_model.parameters()):
            device = params.device
            break
        # for classname in tqdm(classnames):
        self.class_embedding = get_zeroshot_weight(clip_model, openai_imagenet_template, classnames).cpu()
            # classname = tokenize(classname).to(device)
            # self.class_embedding.append(clip_model.encode_text(classname).to('cpu'))

        # self.class_embedding = get_zeroshot_weight(clip_model, openai_imagenet_template, cla)
        super().__init__(IMAGE_PATH, transform)
        # load all images into memory
        samples = []
        if mode == "train":
            if os.path.exists("cache/miniImageNet/mini_train.pt"):
                samples = torch.load("cache/miniImageNet/mini_train.pt")
            else:
                for i in tqdm(range(len(self.samples))):
                    sample = self.loader(self.samples[i][0])
                    samples.append(sample)
                torch.save(samples, "cache/miniImageNet/mini_train.pt")
        elif mode == 'val' or mode == 'test':
            if os.path.exists(f"cache/miniImageNet/mini_{mode}.pt"):
                    samples = torch.load(f"cache/miniImageNet/mini_{mode}.pt")
            else:
                for i in tqdm(range(len(self.samples))):
                    sample = self.loader(self.samples[i][0])
                    samples.append(self.transform(sample))
                torch.save(samples, f"cache/miniImageNet/mini_{mode}.pt")

        self.samples = samples
        self.label = self.targets
    
    def __getitem__(self, index: int):
        sample, target = self.samples[index], self.label[index]
        class_embedding = self.class_embedding[target]
        # if train, transform the image, else, have to transform the image in __init__
        if self.mode == "train":
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, class_embedding, target
    


def return_class():
    return miniImageNet

if __name__ == '__main__':
    a = miniImageNet("data/mini_imagenet/mini_imagenet_split", "val")
    a[1]