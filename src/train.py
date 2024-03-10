import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.shipnet import Shipnet
from src.data.ship_datamodule import ShipDataModule

def main():
    wandb_logger = WandbLogger(project='shipnet', offline=False)

    root_dir = '/home/dralston/shipnet'

    data = {
        'data_path': root_dir + '/data/shipsnet',
        'batch_size': 32,
        'train_val_test_split': [0.7, 0.15, 0.15],
        'transform': None,
    }

    model = {
        'lr': 1e-4,
        'model': 'test_cnn',
    }

    trainer = {
        'max_epochs': 100,
        'devices': 1,
        'callbacks': [ModelCheckpoint(
            monitor='val_loss',
            dirpath=root_dir + '/logs/' + wandb_logger.experiment.name + '/checkpoints',
            filename='shipnet-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )],
    }

    data_module = ShipDataModule(**data)
    model = Shipnet(**model)
    trainer = pl.Trainer(logger=wandb_logger, **trainer)
    wandb_logger.experiment.config.update(data, allow_val_change=True)
    wandb_logger.experiment.config.update(model, allow_val_change=True)
    wandb_logger.experiment.config.update(trainer, allow_val_change=True)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
