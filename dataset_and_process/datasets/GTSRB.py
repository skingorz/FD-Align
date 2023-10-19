from dataset_and_process.datasets.general_dataset import general_dataset

class GTSRB(general_dataset):
    def __init__(self, root="data/meta-dataset/GTSRB/Final_Training/Images", transform=None):
        super().__init__(root, transform)