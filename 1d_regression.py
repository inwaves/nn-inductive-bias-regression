import time

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scipy.interpolate import CubicSpline
from datasets.dataset import glue_dataset_portions
from utils.maths import normalise_data
from utils.utils import adjust_data_linearly, calculate_spline_vs_model_error, setup
from utils.plotting import plot_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make deterministic for reproducibility, or comment out to average runs.
# pl.seed_everything(1337)


if __name__ == '__main__':
    train_dataloader, test_dataloader, data, raw_data, args, train_linreg_pred, test_linreg_pred, model, fn = setup()

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

    # Model is fit to the normalised, linearly adjusted data.
    trainer.fit(model, train_dataloader)
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    # trainer.test(model=model, dataloaders=[test_dataloader])

    # Using raw data...
    raw_x_all, raw_y_all = glue_dataset_portions(raw_x_train, raw_y_train, raw_x_test, raw_y_test)

    # ...generate a grid with more datapoints
    grid = np.linspace(np.min(raw_x_all), np.max(raw_x_all), 100)

    # ...fit the cubic spline.
    spline = CubicSpline(x_train, y_train)

    model = model.to(device)

    # Using adjusted data...
    x_all, y_all = glue_dataset_portions(x_train, y_train, x_test, y_test)
    _, linreg_all = glue_dataset_portions(x_train, train_linreg_pred, x_test, test_linreg_pred)
    linreg_all = linreg_all.reshape(-1, 1)
    all_data = torch.tensor(x_all).float().unsqueeze(1)

    # ...find NN predictions
    y_all_pred = model(all_data).cpu().detach().numpy()  # Using the training and test datapoints.
    # y_all_pred = model(grid).cpu().detach().numpy() # Using a grid that spans the interval of the datapoints.

    # Calculate the difference between g* and the NN function on the grid.
    normalised_grid, [] = normalise_data(grid, [])
    spline_predictions = spline(normalised_grid)
    model_predictions = model(torch.tensor(normalised_grid).float().unsqueeze(1)).cpu().detach().numpy()
    error = calculate_spline_vs_model_error(spline_predictions, model_predictions)

    # Log locally, so I can actually plot these values later...
    with open("logs/nn_vs_variational_solution_error.txt", "a") as f:
        f.write(f"{str(args.hidden_units)}, {str(error)}\n")

    wandb.log({"nn_vs_solution_error": error})

    # Apply ground truth function to the inputs on the grid.
    fn_y = [fn(el) for el in grid]
    _, linreg_spline = adjust_data_linearly(normalised_grid, fn_y)

    # Plot the predictions in the original, non-adjusted, non-normalised space.
    plot_data_vs_predictions(raw_x_train, raw_y_train, raw_x_test, raw_y_test,
                             raw_x_all, y_all_pred+linreg_all, grid, spline_predictions+linreg_spline, fn_y)

    # Wrap up any hanging logger.
    wandb.finish()
