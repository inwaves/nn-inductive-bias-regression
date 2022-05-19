import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from datasets.dataset import glue_dataset_portions
from utils.data_adjuster import DataAdjuster
from utils.parsers import parse_bool, parse_nonlinearity, parse_optimiser, parse_schedule
from utils.maths import mean_squared_error

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShallowNetwork(pl.LightningModule):
    def __init__(self,
                 da_train,
                 da_test,
                 fn,
                 adjust_data_linearly,
                 normalise,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3,
                 nonlinearity="relu",
                 optimiser=None,
                 schedule="none") -> None:
        super().__init__()

        self.da_train = da_train
        self.da_test = da_test

        self.save_hyperparameters()

        self.fn = fn
        self.adjust_data_linearly = adjust_data_linearly
        self.normalise = normalise
        self.lr = lr
        self.hidden = nn.Linear(input_dim, n)
        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.out = nn.Linear(n, output_dim, bias=False)

        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)

    def forward(self, x):
        x = x.to(device)
        return self.out(self.nonlinearity(self.hidden(x)))

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
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3,
                 nonlinearity="relu",
                 optimiser=None,
                 schedule="none") -> None:
        super().__init__()

        self.da_grid, self.fn_y = self.initialise_grid(copy.copy(da_train), copy.copy(da_test), fn, adjust_data_linearly, normalise, grid_resolution)

        self.save_hyperparameters()
        self.lr = lr
        self.k=1000

        # Initialise hidden layers with uniform weights.
        self.hidden1 = nn.Linear(input_dim, n)
        self.hidden1.weight.data.uniform_(-1, 1)
        self.hidden1.bias.data.uniform_(-2, 2)
        self.hidden1.bias.data = self.hidden1.bias.data.to(device)
        self.hidden1.weight.data = self.hidden1.weight.data.to(device)

        self.hidden2 = nn.Linear(input_dim, n)
        self.hidden2.weight.data.uniform_(-1, 1)
        self.hidden2.bias.data.uniform_(-2, 2)
        self.hidden2.weight.data = self.hidden1.weight.data
        self.hidden2.bias.data = self.hidden1.bias.data

        self.hidden1 = self.hidden1.to(device)
        self.hidden2 = self.hidden2.to(device)

        self.nonlinearity = parse_nonlinearity(nonlinearity)

        # Initialise output layers with uniform weights.
        self.out1 = nn.Linear(n, output_dim, bias=False)
        self.out1.weight.data.uniform_(-1, 1)
        self.out1.weight.data = torch.sqrt(torch.tensor(1 / n).to(device)) * self.out1.weight.data.to(device)

        self.out2 = nn.Linear(n, output_dim, bias=False)
        self.out2.weight.data.uniform_(-1, 1)
        self.out2.weight.data = -self.out1.weight.data

        self.optimiser = optimiser(self.parameters(), lr=self.lr)
        self.schedule = parse_schedule(schedule, self.optimiser)
        print(f"In ASI RELU the schedule is: {self.schedule}")

    def initialise_grid(self, da_train, da_test, fn, adjust_data_linearly, normalise, grid_resolution):
        # Using raw data...
        if parse_bool(adjust_data_linearly):
            da_train.unadjust()
            da_test.unadjust()
        if parse_bool(normalise):
            da_train.unnormalise()
            da_test.unnormalise()
        raw_x_all, raw_y_all = glue_dataset_portions(da_train.x, da_train.y, da_test.x, da_test.y)

        # ...generate a grid with more datapoints
        grid = np.linspace(np.min(raw_x_all), np.max(raw_x_all), grid_resolution)
        fn_y = np.array([fn(el) for el in grid]).reshape(1, -1).squeeze()
        da_grid = DataAdjuster(grid, fn_y, da_train.x_min, da_train.x_max)
        if parse_bool(normalise):
            da_grid.normalise()
        return da_grid, fn_y

    def forward(self, x):
        x = x.to(device)

        path1 = self.out1(self.nonlinearity(self.hidden1(x))).to(device)
        path2 = self.out2(self.nonlinearity(self.hidden2(x))).to(device)

        return (torch.sqrt(torch.tensor([2]).to(device)) / 2) * path1 + (
                torch.sqrt(torch.tensor([2]).to(device)) / 2) * path2

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.k != 0:
            return

        model_predictions = self.forward(torch.tensor(self.da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()
        val_loss = mean_squared_error(self.fn_y, model_predictions)
        self.log("val_loss", val_loss)
        return val_loss

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
