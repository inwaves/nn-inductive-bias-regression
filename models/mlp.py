import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.interpolate import CubicSpline

from utils.model_utils import initialise_grid
from utils.maths import infinity_norm_error, mean_squared_error
from utils.parsers import parse_loss_fn, parse_nonlinearity, parse_optimiser, parse_schedule

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(pl.LightningModule):
    def __init__(self,
                 da_train,
                 da_test,
                 fn,
                 adjust_data_linearly,
                 normalise,
                 grid_resolution,
                 hidden_units,
                 input_dim,
                 output_dim,
                 lr=1e-1,
                 nonlinearity="relu",
                 optimiser=None,
                 schedule="none",
                 init="uniform",
                 a_w=1,
                 a_b=2,
                 mu=0,
                 sigma=1,
                 loss="mse") -> None:
        super().__init__()

        da_train = copy.copy(da_train)
        da_test = copy.copy(da_test)
        self.spline = CubicSpline(da_train.x, da_train.y)
        self.da_grid = initialise_grid(da_train, da_test, fn,
                                       adjust_data_linearly, normalise, grid_resolution)

        self.save_hyperparameters()
        self.lr = lr
        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.loss_fn = parse_loss_fn(loss)
        self.input = nn.Linear(input_dim, int(2 / 7 * hidden_units))
        self.hidden1 = nn.Linear(int(2 / 7 * hidden_units), int(3 / 7 * hidden_units))
        self.hidden2 = nn.Linear(int(3 / 7 * hidden_units), int(2 / 7 * hidden_units))

        if init.lower() == "uniform":
            self.hidden1.weight.data.uniform_(-a_w, a_w)
            self.hidden1.bias.data.uniform_(-a_b, a_b)
            self.hidden2.weight.data.uniform_(-a_w, a_w)
            self.hidden2.bias.data.uniform_(-a_b, a_b)
        elif init.lower() == "normal":
            self.hidden1.weight.data.normal_(mu, sigma)
            self.hidden1.bias.data.normal_(mu, sigma)
            self.hidden2.weight.data.normal_(mu, sigma)
            self.hidden2.bias.data.normal_(mu, sigma)

        self.output = nn.Linear(int(2 / 7 * hidden_units), output_dim)
        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)

    def forward(self, x):
        x = x.to(device)
        x = self.nonlinearity(self.input(x))
        x = self.nonlinearity(self.hidden1(x))
        x = self.nonlinearity(self.hidden2(x))
        return self.output(x).to(device)

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        model_predictions = self.forward(torch.tensor(self.da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()

        # Compute validation error (model vs. ground truth).
        val_error = mean_squared_error(self.da_grid.y, model_predictions)
        self.log("validation_error", val_error)

        # Compute variational error (model vs. cubic spline).
        spline_predictions = self.spline(self.da_grid.x)
        variational_error = infinity_norm_error(spline_predictions, model_predictions)
        self.log("variational_error", variational_error)
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
                            "scheduler": self.schedule,
                            "interval":  "epoch",
                            "frequency": 100,
                            "monitor":   "train_loss",
                    }}
        return {"optimizer": self.optimiser}
