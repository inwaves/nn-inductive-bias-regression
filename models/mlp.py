import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.model_utils import initialise_grid
from utils.maths import mean_squared_error
from utils.parsers import parse_nonlinearity, parse_optimiser, parse_schedule

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(pl.LightningModule):
    def __init__(self,
                 da_train,
                 da_test,
                 fn,
                 adjust_data_linearly,
                 normalise,
                 grid_resolution,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-1,
                 nonlinearity="relu",
                 optimiser=None,
                 schedule="none") -> None:
        super().__init__()

        self.da_grid = initialise_grid(copy.copy(da_train), copy.copy(da_test), fn,
                                       adjust_data_linearly, normalise, grid_resolution)

        self.save_hyperparameters()
        self.lr = lr
        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.input = nn.Linear(input_dim, int(2/7*n))
        self.hidden = nn.Sequential(
            nn.Linear(int(2/7 * n), int(3/7 * n)),
            self.nonlinearity,
            nn.Linear(int(3/7 * n), int(2/7 * n)),
            self.nonlinearity,
        )
        self.output = nn.Linear(int(2 / 7 * n), output_dim)
        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)

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

    def validation_step(self, batch, batch_idx):
        model_predictions = self.forward(torch.tensor(self.da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()
        val_error = mean_squared_error(self.da_grid.y, model_predictions)
        self.log("val_error", val_error)
        return val_error

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.schedule is not None:
            return {
                    "optimizer":    self.optimiser,
                    "lr_scheduler": {
                            "scheduler": self.lr_schedule,
                            "interval":  "epoch",
                            "frequency": 100,
                            "monitor":   "train_loss",
                    }}
        return {"optimizer": self.optimiser}
