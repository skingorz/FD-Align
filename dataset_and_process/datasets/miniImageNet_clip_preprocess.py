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
        
        samples = []
        if mode == "train":
            path = "/space0/songk/cache/miniImageNet/mini_train.pkl" if os.path.exists(f"/space0/songk/cache/miniImageNet/mini_train.pkl") else "cache/miniImageNet/mini_train.pkl"
            if os.path.exists(path):
                print(f"loading {path}")
                # load from pickle
                samples = pickle.load(open(path, "rb"))
                # samples = torch.load("cache/miniImageNet/mini_train.pt")
            else:
                for i in tqdm(range(len(self.samples))):
                    sample = self.loader(self.samples[i][0])
                    samples.append(sample)
                # save the samples using pickle
                pickle.dump(samples, open("cache/miniImageNet/mini_train.pkl", "wb"))
                # torch.save(samples, "cache/miniImageNet/mini_train.pt")
        elif mode == 'val' or mode == 'test':
            path = f"/space0/songk/cache/miniImageNet/mini_{mode}.pkl" if os.path.exists(f"/space0/songk/cache/miniImageNet/mini_{mode}.pkl") else f"cache/miniImageNet/mini_{mode}.pkl" 
            if os.path.exists(path):
                    print(f"loading {path}")
                    # load from pickle
                    samples = pickle.load(open(path, "rb"))
                    # samples = torch.load(f"cache/miniImageNet/mini_{mode}.pt")
            else:
                for i in tqdm(range(len(self.samples))):
                    sample = self.loader(self.samples[i][0])
                    samples.append(self.transform(sample))
                # save the samples using pickle
                pickle.dump(samples, open(f"cache/miniImageNet/mini_{mode}.pkl", "wb"))
                # torch.save(samples, f"cache/miniImageNet/mini_{mode}.pt")

        self.samples = samples
        self.label = self.targets

    

    def __getitem__(self, index: int):
        sample, target = self.samples[index], self.label[index]
        # if train, transform the image, else, have to transform the image in __init__
        if self.mode == "train":
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def return_class():
    return miniImageNet

if __name__ == '__main__':
    val = miniImageNet("data/mini_imagenet/mini_imagenet_split", "val")
    train = miniImageNet("data/mini_imagenet/mini_imagenet_split", "train")
    test = miniImageNet("data/mini_imagenet/mini_imagenet_split", "test")
