import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.mlp import MLP
from models.shallow_relu import ShallowRelu, AsiShallowRelu
from utils.plotting import plot_data


# Constants.
device = "cpu"
n = 10

# TODO: solve variational problem over the domain, calculate infinity norm of difference.
# TODO: generate multiple datasets


def generate_sine_wave(gap_size, num_samples):
    """Generate data points for a sine wave where the
    training set ranges [-2π, gap_size) u [2π-gap_size, 4π),
    and the test set ranges [-2π, 4π)."""

    x_train = np.concatenate((np.linspace(-2 * np.pi, gap_size, num_samples//2), np.linspace(2 * np.pi - gap_size, 4 * np.pi, num_samples//2)))
    x_test = np.linspace(gap_size, 2*np.pi-gap_size, num_samples//2)
    y_train = np.sin(x_train)
    y_test = np.sin(x_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    pl.seed_everything(1337)

    x_train, y_train, x_test, y_test = generate_sine_wave(gap_size=0, num_samples=10)
    training_data = np.array(list(zip(x_train, y_train)))
    test_data = np.array(list(zip(x_test, y_test)))

    # We're doing full-batch gradient descent, so the batch_size = n
    train_dataloader = DataLoader(training_data, batch_size=len(x_train))
    test_dataloader = DataLoader(test_data, batch_size=len(x_test))

    model = AsiShallowRelu(n, 1, 1).to(device).float()
    # model = ShallowRelu(n, 1, 1).to(device).float()
    # model = MLP(n, 1, 1).to(device).float()

    early_stopping_callback = EarlyStopping(monitor="train_loss", min_delta=1e-8, patience=3)
    trainer = pl.Trainer(max_epochs=-1 ,
                         callbacks=[early_stopping_callback],
                         log_every_n_steps=1,)
    trainer.fit(model, train_dataloader)

    trainer.test(model=model, dataloaders=[test_dataloader])
    x_all = np.concatenate([x_train, x_test])
    x_all.sort()

    y_pred = model(torch.tensor(x_all).float().unsqueeze(1).to(device)).detach().numpy()
    plot_data(x_train, y_train, x_test, y_test, y_pred, x_all)

