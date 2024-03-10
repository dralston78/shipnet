import glob
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.components.ship_dataset import ShipDataset

class ShipDataModule(LightningDataModule):
    def __init__(
            self,
            data_path,
            batch_size=32,
            train_val_test_split=[0.7, 0.15, 0.15],
            transform=None
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage='fit'):
        # Get a list of all image files in the directory
        all_files = glob.glob(self.data_path + '/*.png')

        #use random split 
        train_files, test_files = train_test_split(all_files, test_size=self.train_val_test_split[2], random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=self.train_val_test_split[1], random_state=42)

        # Create ShipDataset instances for the train, validation, and test sets
        self.train_dataset = ShipDataset(train_files, self.data_path, transform=self.transform)
        self.val_dataset = ShipDataset(val_files, self.data_path, transform=self.transform)
        self.test_dataset = ShipDataset(test_files, self.data_path, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=11,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            persistent_workers=True,
        )
    