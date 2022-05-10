import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.parsers import parse_nonlinearity, parse_optimiser

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-1,
                 nonlinearity="relu",
                 optimiser="sgd") -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.optimiser = optimiser
        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.input = nn.Linear(input_dim, int(2/7*n))
        self.hidden = nn.Sequential(
            nn.Linear(int(2/7 * n), int(3/7 * n)),
            self.nonlinearity,
            nn.Linear(int(3/7 * n), int(2/7 * n)),
            self.nonlinearity,
        )
        self.output = nn.Linear(int(2 / 7 * n), output_dim)

    def forward(self, x):
        x = x.to(device)
        x = self.nonlinearity(self.input(x))
        return self.output(self.hidden(x)).to(device)

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = parse_optimiser(self.optimiser)(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }
