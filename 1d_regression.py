import time

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scipy.interpolate import CubicSpline
from datasets.dataset import glue_dataset_portions
from utils.data_adjuster import DataAdjuster
from utils.maths import linear, normalise_data
from utils.utils import calculate_spline_vs_model_error, parse_bool, setup
from utils.plotting import plot_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make deterministic for reproducibility, or comment out to average runs.
# pl.seed_everything(1337)


if __name__ == '__main__':
    train_dataloader, test_dataloader, da_train, da_test, args, model, fn = setup()

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

    # Model is fit to the normalised, linearly adjusted data.
    tic = time.time()
    trainer.fit(model, train_dataloader)
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    trainer.test(model=model, dataloaders=[test_dataloader])

    # Using raw data...
    raw_x_all, raw_y_all = glue_dataset_portions(raw_x_train, raw_y_train, raw_x_test, raw_y_test)

    # ...generate a grid with more datapoints
    grid = np.linspace(np.min(raw_x_all), np.max(raw_x_all), 100)
    da_grid = DataAdjuster(grid, None)
    if args.normalise:
        da_grid.normalise()

    model = model.to(device)

    # Using adjusted data...
    # ...fit the cubic spline.
    spline = CubicSpline(da_train.x, da_train.y)

    # ...remembering the linear adjustment, so we can undo it when plotting
    x_all, y_all = glue_dataset_portions(da_train.x, da_train.y, da_test.x, da_test.y)
    if parse_bool(args.adjust_data_linearly):
        da_train.unadjust()
        da_test.unadjust()
    all_data = torch.tensor(x_all).float().unsqueeze(1)

    # ...find NN predictions
    y_all_pred = model(all_data).cpu().detach().numpy()  # Using the training and test datapoints.

    # Calculate the difference between g* and the NN function on the grid.
    spline_predictions = spline(da_grid.x)
    model_predictions = model(torch.tensor(da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()
    error = calculate_spline_vs_model_error(spline_predictions, model_predictions)

    # Log locally, so I can actually plot these values later...
    with open("logs/nn_vs_variational_solution_error.txt", "a") as f:
        f.write(f"{args.dataset}-{args.generalisation_task}, {str(args.hidden_units)}, {str(error)}\n")

    wandb.summary["nn_vs_solution_error"] = error

    # Apply ground truth function to the inputs on the grid.
    fn_y = np.array([fn(el) for el in grid]).reshape(1, -1).squeeze()

    unadjusted_spline_preds = spline_predictions + linear(normalised_grid, linear_fit.intercept_, linear_fit.coef_[0]) \
        if parse_bool(args.adjust_data_linearly) else spline_predictions
    # Plot the predictions in the original, non-adjusted, non-normalised space.
    plot = plot_data_vs_predictions(raw_x_train, raw_y_train, raw_x_test, raw_y_test,
                                    raw_x_all, y_all_pred + linreg_all, grid,
                                    unadjusted_spline_preds, fn_y, args)

    # Wrap up any hanging logger.
    wandb.finish()
