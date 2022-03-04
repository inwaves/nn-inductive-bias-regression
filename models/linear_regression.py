import pytorch_lightning
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LinearRegression(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.save_hyperparameters()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-1)
