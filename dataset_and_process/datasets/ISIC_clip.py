from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
from architectures.feature_extractor.clip import load

class ISIC(Dataset):
    def __init__(self, root="data/meta-dataset/ISIC", mode="test", backbone_name="resnet12", transform=None):
        """
        Args:
            root: Root directory path.
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = os.path.join(root, "ISIC2018_Task3_Training_Input")
        self.csv_path = os.path.join(root, "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
        _, train_process, val_process=load(backbone_name, jit=False)
        if mode == 'val' or mode == 'test':
            transform = val_process
        elif mode == 'train':
            transform = train_process
        

        self.transform = transform


        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.label = np.asarray(self.data_info.iloc[:, 1:])

        # print(self.labels[:10])
        self.label = (self.label!=0).argmax(axis=1)

        # Calculate len
        self.data_len = len(self.label)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        image = Image.open(os.path.join(self.img_path, single_image_name + ".jpg")).convert('RGB')
        image = self.transform(image)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label[index]

        return (image, single_image_label)

    def __len__(self):
        return self.data_len

def return_class():
    return ISIC