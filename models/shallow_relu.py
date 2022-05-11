import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.parsers import parse_nonlinearity, parse_optimiser

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShallowNetwork(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3,
                 nonlinearity="relu",
                 optimiser="sgd") -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.optimiser = optimiser
        self.hidden = nn.Linear(input_dim, n)
        self.nonlinearity = parse_nonlinearity(nonlinearity)
        self.out = nn.Linear(n, output_dim, bias=False)

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
        optimizer = parse_optimiser(self.optimiser)(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                # patience=2),
        }


class AsiShallowNetwork(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3,
                 nonlinearity="relu",
                 optimiser="sgd") -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.optimiser = optimiser

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

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1).to(device), batch[:, 1].float().unsqueeze(1).to(device)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = parse_optimiser(self.optimiser)(self.parameters(), lr=self.lr)
        return {
                "optimizer":    optimizer,
                "lr_scheduler": {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                mode="min",
                                                                                factor=0.5,
                                                                                patience=10,
                                                                                verbose=True),
                        "monitor":   "train_loss",
                }}


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
