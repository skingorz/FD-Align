from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
from architectures.feature_extractor.clip import load

class OxfordFlowers102Dataset(Dataset):
    """
    Oxford 102 Category Flower Dataset
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    """

    def __init__(self, root="data/meta-dataset/VGGFlower", mode="test", backbone_name="resnet12", transform=None):

        self.root = root
        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        self.transform = transform



        labels_filename = self.root + "/imagelabels.mat"
        # shift labels from 1-index to 0-index
        self.label = loadmat(labels_filename)["labels"].flatten() - 1

    def __getitem__(self, index):
        filepath = self.root + "/jpg" + f"/image_{index+1:05}.jpg"
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.labels)
    
def return_class():
    return OxfordFlowers102Dataset