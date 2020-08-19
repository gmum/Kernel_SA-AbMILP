import pandas as pd
import numpy as np
from PIL import Image
from typing import List
from torch.utils.data import Dataset

from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor

normalization_mean = [89.7121552586411, 89.7121552586411, 89.7121552586411]
normalization_std = [18.49568745464706, 15.415668522447366, 11.147622232506315]


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, patches_file: str, label_file: str, transforms: List):
        self.df_patches = pd.read_csv(patches_file)
        self.df_labels = pd.read_csv(label_file)
        self.df_labels = self.df_labels[self.df_labels['image'].isin(self.df_patches.image.values)]
        transforms = [*transforms, ToTensor()]
        transforms.append(Normalize(mean=normalization_mean, std=normalization_std))
        self.transforms = Compose(transforms)

        self.bags, self.labels = self.create_bags()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def create_bags(self):
        labels, bags = [], []
        for i, row in self.df_labels.iterrows():
            label = row['level']
            img_id = row['image']
            curr_paths = self.df_patches[self.df_patches['image'] == img_id]['path'].values
            curr_bag = []
            for path in curr_paths:
                curr_bag.append(self.transforms(np.asarray(Image.open(path))))
            labels.append(label)
            bags.append(curr_bag)
        return bags, labels
