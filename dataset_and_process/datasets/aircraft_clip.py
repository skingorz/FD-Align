import os
from PIL import Image
from torch.utils.data import Dataset
from architectures.feature_extractor.clip import load

FILENAME_LENGTH = 7


class AircraftDataset(Dataset):
    """
    Args:
        root: Root directory path.
        transform: pytorch transforms for transforms and tensor conversion
    """

    def __init__(self, root="data/meta-dataset/fgvc-aircraft-2013b/data", mode="test", backbone_name="resnet12", transform=None):
        self.root = root
        variants_dict = {}
        with open(os.path.join(root, 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                variants_dict[line.strip()] = idx
        self.num_classes = len(variants_dict)


        list_path = os.path.join(root, 'images_variant_test.txt')

        self.images = []
        self.label = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                fname_and_variant = line.strip()
                self.images.append(fname_and_variant[:FILENAME_LENGTH])
                self.label.append(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]])

        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == "train":
            self.transform = train_process
        else:
            self.transform = val_process

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.root, 'images', '%s.jpg' % self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.label[item]  # count begin from zero

    def __len__(self):
        return len(self.images)
    
def return_class():
    return AircraftDataset