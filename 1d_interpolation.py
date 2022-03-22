import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scipy.interpolate import CubicSpline
from datasets.dataset import generate_sine_interpolation_dataset, glue_dataset_portions, \
    generate_sine_extrapolation_dataset
from models.mlp import MLP
from models.linear_regression import LinearRegression
from models.shallow_relu import AsiShallowRelu, ShallowRelu
from utils.plotting import plot_sin_data_vs_predictions

# TODO: cherry pick the samples instead of randomly generating every time

# Constants.
device = "cuda" if torch.cuda.is_available() else "cpu"
n = 800
num_samples = 50


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

    x_train, y_train, x_test, y_test = generate_sine_interpolation_dataset(gap_size=np.pi/2, num_train_datapoints=num_samples)
    # x_train, y_train, x_test, y_test = generate_sine_extrapolation_dataset(num_train_datapoints=num_samples,
    #                                                                        num_test_datapoints=num_samples//2)

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
    trainer = pl.Trainer(max_epochs=-1,
                         callbacks=[early_stopping_callback],
                         log_every_n_steps=1,)
    trainer.fit(model, train_dataloader)

    trainer.test(model=model, dataloaders=[test_dataloader])
    print(x_train, x_test)

    # Apply g* to all the data points.
    x_all, y_all = glue_dataset_portions(x_train, y_train, x_test, y_test)
    grid = np.linspace(np.min(x_all), np.max(x_all), 100)
    spline = CubicSpline(x_train, y_train)

    y_pred = model(torch.tensor(x_all).float().unsqueeze(1).to(device)).detach().numpy()
    plot_sin_data_vs_predictions(x_train, y_train, x_test, y_test, y_pred, x_all, grid, spline(grid))

