import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ShipDataset(Dataset):
    def __init__(self, files, data_path, transform=None):
        self.filenames = files
        self.labels = [self._get_label(filename) for filename in self.filenames if self._get_label(filename) is not None]
        self.data_path = data_path
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.size=4000

    def __len__(self):
        return min(self.size, len(self.filenames))
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_path, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.to_tensor(image)
        label = self.labels[idx]
        label = torch.tensor(label)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label


    def _get_label(self, filename):
        label = filename.split('__')
        if len(label) == 3:
            return int(label[0][-1])
        else: 
            return None 