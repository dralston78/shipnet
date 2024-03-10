
import torch 
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.components.steerable_cnn import SO2SteerableCNN
from src.models.components.test_cnn import CNN

class Shipnet(LightningModule):
    def __init__(self,
        lr = 1e-4,
        model = 'steerable_cnn'
    ):
        super().__init__()

        self.lr = lr
        if model == 'steerable_cnn':
            self.model = SO2SteerableCNN()
        elif model == 'test_cnn':
            self.model = CNN()
        else:
            raise KeyError

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        pass 

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        # Training step for one batch.
        x, y = batch
        logits = self.model(x)
        logits, y = logits.squeeze().float(), y.squeeze().float()
        loss = F.binary_cross_entropy_with_logits(logits, y)  # Use BCEWithLogitsLoss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step for one batch.
        x, y = batch
        logits = self.model(x)
        logits, y = logits.squeeze().float(), y.squeeze().float()
        loss = F.binary_cross_entropy_with_logits(logits, y)  # Use BCEWithLogitsLoss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Test step for one batch.
        x, y = batch
        logits = self.model(x)
        logits, y = logits.squeeze().float(), y.squeeze().float()
        loss = F.binary_cross_entropy_with_logits(logits, y)  # Use BCEWithLogitsLoss
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)