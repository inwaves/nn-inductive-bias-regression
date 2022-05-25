import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.interpolate import CubicSpline

from utils.model_utils import initialise_grid
from utils.parsers import parse_loss_fn, parse_nonlinearity, parse_schedule
from utils.maths import mean_squared_error

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShallowNetwork(pl.LightningModule):
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
                 lr=1e-3,
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

        # This saves all the hparams in the logger.
        self.save_hyperparameters()

        self.lr = lr
        self.hidden = nn.Linear(input_dim, hidden_units)

        if init.lower() == "uniform":
            self.hidden.weight.data.uniform_(-a_w, a_w)
            self.hidden.bias.data.uniform_(-a_b, a_b)
        elif init.lower() == "normal":
            self.hidden.weight.data.normal_(mu, sigma)
            self.hidden.bias.data.normal_(mu, sigma)
        self.out = nn.Linear(hidden_units, output_dim, bias=False)

        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.loss_fn = parse_loss_fn(loss)

        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)

    def forward(self, x):
        x = x.to(device)
        return self.out(self.nonlinearity(self.hidden(x)))

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        model_predictions = self.forward(torch.tensor(self.da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()

        # Compute validation error (model vs. ground truth).
        validation_error = mean_squared_error(self.da_grid.y, model_predictions)
        self.log("validation_error", validation_error)

        # Compute variational error (model vs. cubic spline).
        spline_predictions = self.spline(self.da_grid.x)
        variational_error = mean_squared_error(spline_predictions, model_predictions)
        self.log("variational_error", variational_error)

        return validation_error

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


class AsiShallowNetwork(pl.LightningModule):
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
                 lr=1e-3,
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

        # Initialise hidden layers with uniform weights.
        self.hidden1 = nn.Linear(input_dim, hidden_units)
        self.hidden2 = nn.Linear(input_dim, hidden_units)

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

        self.hidden1.bias.data = self.hidden1.bias.data.to(device)
        self.hidden1.weight.data = self.hidden1.weight.data.to(device)
        self.hidden2.weight.data = self.hidden1.weight.data
        self.hidden2.bias.data = self.hidden1.bias.data

        self.hidden1 = self.hidden1.to(device)
        self.hidden2 = self.hidden2.to(device)

        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.loss_fn = parse_loss_fn(loss)

        # Initialse output layers with uniform weights.
        self.out1 = nn.Linear(hidden_units, output_dim, bias=False)
        self.out1.weight.data.uniform_(-1, 1)
        self.out1.weight.data = torch.sqrt(torch.tensor(1 / hidden_units).to(device)) * self.out1.weight.data.to(device)

        self.out2 = nn.Linear(hidden_units, output_dim, bias=False)
        self.out2.weight.data.uniform_(-1, 1)
        self.out2.weight.data = -self.out1.weight.data

        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)
        print(f"In ASI RELU the schedule is: {self.schedule}")

    def forward(self, x):
        x = x.to(device)

        path1 = self.out1(self.nonlinearity(self.hidden1(x))).to(device)
        path2 = self.out2(self.nonlinearity(self.hidden2(x))).to(device)

        return (torch.sqrt(torch.tensor([2]).to(device)) / 2) * path1 + (
                torch.sqrt(torch.tensor([2]).to(device)) / 2) * path2

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
        variational_error = mean_squared_error(spline_predictions, model_predictions)
        self.log("variational_error", variational_error)
        return val_error

    def configure_optimizers(self):
        if self.schedule is not None:
            return {
                    "optimizer":    self.optimiser,
                    "lr_scheduler": {
                            "scheduler": self.schedule,
                            "interval": "epoch",
                            "frequency": 100,
                            "monitor":   "train_loss",
                    }}
        return {"optimizer":    self.optimiser}


class PlainTorchAsiShallowRelu(nn.Module):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 nonlinearity) -> None:
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, n)
        self.hidden2 = nn.Linear(input_dim, n)
        self.hidden2.weight.data = self.hidden1.weight.data
        self.hidden2.bias.data = self.hidden1.bias.data
        self.nonlinearity = parse_nonlinearity(nonlinearity)

        self.out1 = nn.Linear(n, output_dim, bias=False)
        self.out1.weight.data = torch.sqrt(torch.tensor(1 / n).to(device)) * self.out1.weight.data

        self.out2 = nn.Linear(n, output_dim, bias=False)
        self.out2.weight.data = -self.out1.weight.data

    def forward(self, x):
        x = x.to(device)
        path1 = self.out1(self.nonlinearity(self.hidden1(x)))
        path2 = self.out2(self.nonlinearity(self.hidden2(x)))
        return (torch.sqrt(torch.tensor([2]).to(device)) / 2) * path1 + (
                torch.sqrt(torch.tensor([2]).to(device)) / 2) * path2
