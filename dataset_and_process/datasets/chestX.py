from dataset_and_process.datasets.general_dataset import general_dataset

class ChestX(general_dataset):
    def __init__(self, root="data/meta-dataset/ChestX/data", transform=None):
        super().__init__(root, transform)