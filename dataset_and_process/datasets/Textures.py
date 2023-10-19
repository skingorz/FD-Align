from dataset_and_process.datasets.general_dataset import general_dataset

class Textures(general_dataset):
    def __init__(self, root="data/meta-dataset/Textures/dtd/images", transform=None):
        super().__init__(root, transform)