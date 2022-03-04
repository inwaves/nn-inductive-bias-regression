import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datasets.dataset import generate_sine_interpolation_dataset
from models.mlp import MLP
from models.linear_regression import LinearRegression
from models.shallow_relu import AsiShallowRelu
from utils.plotting import plot_sin_data_vs_predictions


# TODO: solve variational problem over the domain, calculate infinity norm of difference.
# TODO: generate multiple datasets

# Constants.
device = "cuda" if torch.cuda.is_available() else "cpu"
n = 1000
num_samples = 10


def adjust_data_linearly(x_train, y_train):
    """Fit a linear regression model."""

    training_data = np.array(list(zip(x_train, y_train)))
    print(f"Training data: {training_data}")
    train_dataloader = DataLoader(training_data, batch_size=len(x_train))

    model = LinearRegression(input_dim=1, output_dim=1).to(device).float()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloader)
    y_hat = model(torch.tensor(x_train).float().unsqueeze(1).to(device))
    residual = y_train - y_hat.detach().numpy().reshape(y_train.shape)

    print(f"Shape of y_hat: {y_hat.detach().numpy().shape}")
    print(f"Shape of residual: {residual.shape}, shape of y_train: {y_train.shape}")
    return residual


if __name__ == '__main__':
    pl.seed_everything(1337)

    x_train, y_train, x_test, y_test = generate_sine_interpolation_dataset(gap_size=0, num_train_datapoints=num_samples)

    # Fit linear regression model.
    # y_train = adjust_data_linearly(x_train, y_train)
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
    y_all = np.sin(x_all)

    y_pred = model(torch.tensor(x_all).float().unsqueeze(1).to(device)).detach().numpy()
    plot_sin_data_vs_predictions(x_train, y_train, x_test, y_test, y_pred, x_all, y_all)

