from dataset_and_process.datasets.general_dataset import general_dataset
import os
from tqdm import tqdm
import pickle
from architectures.feature_extractor.clip import load
class Clipart(general_dataset):
    def __init__(self, root="data/meta-dataset/clipart", mode="test", backbone_name="resnet12", transform=None):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        super().__init__(root, transform)
        self.label = self.targets
        
        # # path = "cache/clippart/clippart.pkl"
        # # if os.path.exists(path):
        # #     samples = pickle.load(open(path, "rb"))
        # # else:
        # samples = []
        # for i in tqdm(range(len(self.samples))):
        #     sample = self.loader(self.samples[i][0])
        #     samples.append(self.transform(sample))
        # # pickle.dump(samples, open(path, "wb"))
        # self.samples = samples
        # self.label = self.targets

    # def __getitem__(self, index: int):
    #     path, target = self.samples[index], self.label[index]
    #     # if train, transform the image, else, have to transform the image in __init__
    #     sample = 
    #     if self.mode == "train":
    #         if self.transform is not None:
    #             sample = self.transform(sample)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return sample, target
def return_class():
    return Clipart

if __name__ == "__main__":
    # val = CUB("data/meta-dataset/cub", "val")
    test = Clipart("data/meta-dataset/cub", "test")
    # train = CUB("data/meta-dataset/cub", "tr")
    