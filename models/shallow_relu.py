import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ShallowRelu(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.hidden = nn.Linear(input_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n, output_dim)

    def forward(self, x):
        return self.out(self.relu(self.hidden(x)))

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
        }
