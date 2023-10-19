from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
import pickle
from architectures.feature_extractor.clip import load

class CUB(Dataset):
    r"""The miniImageNet dataset.
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """
    def __init__(self, root: str, mode: str,  backbone_name="resnet12", transform=None) -> None:
        
        IMAGE_PATH = osp.join(root, 'images')
        SPLIT_PATH = osp.join(root, 'split')
        assert mode in ["train", "val", "test"]
        self.mode = mode
        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process

        csv_path = osp.join(SPLIT_PATH, f'{mode}.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        wnids = []

        # if os.path.exists(f"cache/cub/{mode}.pkl"):
        #     print(f"loading cache/cub/{mode}.pkl")
        #     data, label = pickle.load(open(f"cache/cub/{mode}.pkl", "rb"))
        # else:
        for l in lines:
            name, wnid = l.split(',')[:2]
            path = osp.join(IMAGE_PATH, name)
            # image = Image.open(path).convert('RGB')
            # if mode == "val" or mode == "test":
            #     image = transform(image)
            if wnid not in wnids:
                wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        # pickle.dump((data, label), open(f"cache/cub/{mode}.pkl", "wb"))

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = Image.open(path).convert('RGB')
        # if self.mode == "train":
        image = self.transform(image)
        return image, label

def return_class():
    return CUB

if __name__ == "__main__":
    # val = CUB("data/meta-dataset/cub", "val")
    test = CUB("data/meta-dataset/cub", "test")
    # train = CUB("data/meta-dataset/cub", "tr")
    