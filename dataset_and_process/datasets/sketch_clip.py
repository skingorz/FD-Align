from dataset_and_process.datasets.general_dataset import general_dataset
import os
from tqdm import tqdm
import pickle
from architectures.feature_extractor.clip import load
class sketch(general_dataset):
    def __init__(self, root="data/meta-dataset/real", mode="test", backbone_name="resnet12", transform=None):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        super().__init__(root, transform)
        self.label = self.targets
        
def return_class():
    return sketch

if __name__ == "__main__":
    # val = CUB("data/meta-dataset/cub", "val")
    test = sketch("data/meta-dataset/cub", "test")
    # train = CUB("data/meta-dataset/cub", "tr")
    