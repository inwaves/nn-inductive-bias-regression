import time

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scipy.interpolate import CubicSpline
from datasets.dataset import glue_dataset_portions
from utils.utils import variational_solution_vs_neural_network, setup
from utils.plotting import plot_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make deterministic for reproducibility, or comment out to average runs.
# pl.seed_everything(1337)


if __name__ == '__main__':
    train_dataloader, test_dataloader, data, raw_data, args, model, fn = setup()
    x_train, y_train, x_test, y_test = data
    raw_x_train, raw_y_train, raw_x_test, raw_y_test = raw_data

    early_stopping_callback = EarlyStopping(monitor="train_loss", min_delta=1e-8, patience=3)

    wandb_logger = WandbLogger(project="generalisation")

    # This control flow is needed to be able to run this script
    # on either CPU (locally) or GPU (on a cluster).
    if device == "cuda":
        trainer = pl.Trainer(max_epochs=-1,
                             callbacks=[early_stopping_callback],
                             accelerator="gpu",
                             devices=1,
                             logger=wandb_logger,
                             log_every_n_steps=args.log_every_k_steps, )
    else:
        trainer = pl.Trainer(max_epochs=-1,
                             callbacks=[early_stopping_callback],
                             accelerator="cpu",
                             logger=wandb_logger,
                             log_every_n_steps=args.log_every_k_steps, )

    tic = time.time()
    trainer.fit(model, train_dataloader)
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    # trainer.test(model=model, dataloaders=[test_dataloader])

    x_all, y_all = glue_dataset_portions(raw_x_train, raw_y_train, raw_x_test, raw_y_test)

    # Grid is used to plot g* correctly, otherwise it doesn't match the actual function
    # because there are not that many data points in x_all.
    grid = np.linspace(np.min(x_all), np.max(x_all), 100)

    # Fit the cubic spline to the *adjusted* training data so it matches what the modesl was trained on.
    spline = CubicSpline(x_train, y_train)

    model = model.to(device)
    print(f"Model device: {model.device}")

    # Find NN predictions for all data points (train + test).
    all_data = torch.tensor(x_all).float().unsqueeze(1)
    print(f"all_data.device: {all_data.device}")
    y_all_pred = model(all_data).cpu().detach().numpy()

    # Calculate the difference between g* and the NN function on the training data.
    y_variational = spline(x_train)
    y_train_pred = model(torch.tensor(x_train).float().unsqueeze(1)).cpu().detach().numpy()
    error = variational_solution_vs_neural_network(y_variational, y_train_pred)

    # Log locally, so I can actually plot these values later...
    with open("logs/nn_vs_variational_solution_error.txt", "a") as f:
        f.write(f"{str(args.hidden_units)}, {str(error)}\n")

    wandb.log({"nn_vs_solution_error": error})

    # Plot the predictions in the original, non-adjusted, non-normalised space.
    plot_data_vs_predictions(raw_x_train, raw_y_train, raw_x_test, raw_y_test,
                             y_all_pred, x_all, grid, spline(grid), fn)

    # Wrap up any hanging logger.
    wandb.finish()
