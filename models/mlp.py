import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(pl.LightningModule):
    def __init__(self,
                 n,
                 input_dim,
                 output_dim,
                 lr=1e-1) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, n),
            nn.ReLU(),
            nn.Linear(n, 2 * n),
            nn.ReLU(),
            nn.Linear(2 * n, 3 * n),
            nn.ReLU(),
            nn.Linear(3 * n, 2 * n),
            nn.ReLU(),
            nn.Linear(2 * n, n),
            nn.ReLU(),
        )
        self.out = nn.Linear(n, output_dim)

    def forward(self, x):
        x = x.to(device)
        return self.out(self.hidden(x)).to(device)

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
        }
