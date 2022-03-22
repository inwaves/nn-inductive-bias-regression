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

        self.hidden = nn.Linear(input_dim, n)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n, output_dim, bias=False)
        # self.out.weight.data = torch.sqrt(torch.tensor(1.0 / n)) * self.out.weight.data

    def forward(self, x):
        return self.out(self.relu(self.hidden(x)))

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
        }


class AsiShallowRelu(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.hidden1 = nn.Linear(input_dim, n)
        self.hidden2 = nn.Linear(input_dim, n)
        self.hidden2.weight.data = self.hidden1.weight.data
        self.hidden2.bias.data = self.hidden1.bias.data
        self.relu = nn.ReLU()

        self.out1 = nn.Linear(n, output_dim, bias=False)
        self.out1.weight.data = torch.sqrt(torch.tensor(1/n)) * self.out1.weight.data

        self.out2 = nn.Linear(n, output_dim, bias=False)
        self.out2.weight.data = -self.out1.weight.data

    def forward(self, x):
        path1 = self.out1(self.relu(self.hidden1(x)))
        path2 = self.out2(self.relu(self.hidden2(x)))
        return (torch.sqrt(torch.tensor([2]))/2)*path1 + (torch.sqrt(torch.tensor([2]))/2)*path2

    def training_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        idx, targets = batch[:, 0].float().unsqueeze(1), batch[:, 1].float().unsqueeze(1)
        out = self.forward(idx)

        loss = F.mse_loss(out, targets)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
        }