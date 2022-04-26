import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

device = "cuda" if torch.cuda.is_available() else "cpu"


class ShallowRelu(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.hidden = nn.Linear(input_dim, n)
        self.relu = nn.ReLU()
        self.out = nn.Linear(n, output_dim, bias=False)

    def forward(self, x):
        x = x.to(device)
        return self.out(self.relu(self.hidden(x)))

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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
        }


class AsiShallowRelu(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-3) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

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

        self.relu = nn.ReLU()

        # Initialise output layers with uniform weights.
        self.out1 = nn.Linear(n, output_dim, bias=False)
        self.out1.weight.data.uniform_(-1, 1)
        self.out1.weight.data = torch.sqrt(torch.tensor(1 / n).to(device)) * self.out1.weight.data.to(device)

        self.out2 = nn.Linear(n, output_dim, bias=False)
        self.out2.weight.data.uniform_(-1, 1)
        self.out2.weight.data = -self.out1.weight.data

    def forward(self, x):
        x = x.to(device)

        hid1 = self.hidden1(x)
        hid2 = self.hidden2(x)

        print(f"hid1.device: {hid1.device}")
        print(f"hid1.device: {hid1.device}")

        rel1 = self.relu(hid1)
        rel2 = self.relu(hid2)

        print(f"rel1.device: {rel1.device}")
        print(f"rel2.device: {rel2.device}")

        p1 = self.out1(rel1)
        p2 = self.out2(rel2)

        print(f"p1.device: {p1.device}")
        print(f"p2.device: {p2.device}")

        # path1 = self.out1(self.relu(self.hidden1(x))).to(device)
        # path2 = self.out2(self.relu(self.hidden2(x))).to(device)

        return (torch.sqrt(torch.tensor([2]).to(device)) / 2) * p1 + (
                    torch.sqrt(torch.tensor([2]).to(device)) / 2) * p1

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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2),
        }


class PlainTorchAsiShallowRelu(nn.Module):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim, ) -> None:
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, n)
        self.hidden2 = nn.Linear(input_dim, n)
        self.hidden2.weight.data = self.hidden1.weight.data
        self.hidden2.bias.data = self.hidden1.bias.data
        self.relu = nn.ReLU()

        self.out1 = nn.Linear(n, output_dim, bias=False)
        self.out1.weight.data = torch.sqrt(torch.tensor(1 / n).to(device)) * self.out1.weight.data

        self.out2 = nn.Linear(n, output_dim, bias=False)
        self.out2.weight.data = -self.out1.weight.data

    def forward(self, x):
        x = x.to(device)
        path1 = self.out1(self.relu(self.hidden1(x)))
        path2 = self.out2(self.relu(self.hidden2(x)))
        return (torch.sqrt(torch.tensor([2]).to(device)) / 2) * path1 + (
                    torch.sqrt(torch.tensor([2]).to(device)) / 2) * path2
