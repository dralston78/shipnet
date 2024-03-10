import torch
from src.models.shipnet import Shipnet
from src.data.ship_datamodule import ShipDataModule
from pytorch_lightning import Trainer

def main():
    root_dir = '/home/dralston/shipnet'

    data = {
        'data_path': root_dir + '/data/shipsnet',
        'batch_size': 32,
        'train_val_test_split': [.7,.15,.15],  # All data goes to the test set
        'transform': None,
    }

    model = {
        'lr': 1e-4,
        'model': 'test_cnn',
    }

    trainer = {
        'devices': 1,
    }

    # Load the test data
    data_module = ShipDataModule(**data)

    # Load the trained model
    model_path = root_dir + '/logs/2024-03-09_18-33-43/checkpoints/shipnet-epoch=89-val_loss=0.05.ckpt'  # Update with your model path
    model = Shipnet.load_from_checkpoint(checkpoint_path=model_path, **model)

    # Initialize the trainer
    trainer = Trainer(**trainer)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()