import time
import copy

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.loggers import WandbLogger
from scipy.interpolate import CubicSpline

from datasets.dataset import glue_dataset_portions
from utils.adjust_data import DataAdjuster
from utils.maths import infinity_norm_error, linear, mean_squared_error
from utils.utils import parse_bool, setup
from utils.plotting import plot_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make deterministic for reproducibility, or comment out to average runs.
# pl.seed_everything(1337)


if __name__ == '__main__':
    # Wrap up any hanging logger.
    wandb.finish()
    wandb_logger = WandbLogger(project="gen2")

    train_dataloader, test_dataloader, da_train, da_test, args, \
        model, fn, callbacks, dirpath, early_stopping, max_epochs = setup()

    num_epochs = f"{max_epochs}epochs" if max_epochs != -1 else f"infinite_epochs"
    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=callbacks,
                         accelerator="gpu" if device == "cuda" else "cpu",
                         devices=1 if device == "cuda" else None,
                         logger=wandb_logger,
                         log_every_n_steps=args.log_every_n_steps,
                         check_val_every_n_epoch=args.val_frequency)

    # Model is fit to the normalised, linearly adjusted data.
    tic = time.time()
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=[train_dataloader],
                )

    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    model = model.to(device)

    # Using adjusted data...
    # ...fit the cubic spline.
    x_tr, y_tr, x_te, y_te = copy.copy(da_train.x), copy.copy(da_train.y), copy.copy(da_test.x), copy.copy(da_test.y)
    spline = CubicSpline(x_tr, y_tr)

    # Using raw data...
    if parse_bool(args.adjust_data_linearly):
        da_train.unadjust()
        da_test.unadjust()
    if parse_bool(args.normalise):
        da_train.unnormalise()
        da_test.unnormalise()
    raw_x_all, raw_y_all = glue_dataset_portions(da_train.x, da_train.y, da_test.x, da_test.y)

    # ...generate a grid with more datapoints
    grid = np.linspace(np.min(raw_x_all), np.max(raw_x_all), args.grid_resolution)
    fn_y = np.array([fn(el) for el in grid]).reshape(1, -1).squeeze()
    da_grid = DataAdjuster(grid, fn_y, da_train.x_min, da_train.x_max)
    if parse_bool(args.normalise):
        da_grid.normalise()
    if parse_bool(args.adjust_data_linearly):
        da_grid.adjust()

    # Calculate the final variational error as the difference
    # between g* and the model on the grid.
    spline_predictions = spline(da_grid.x)
    model_predictions = model(torch.tensor(da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()
    variational_error = infinity_norm_error(spline_predictions, model_predictions)

    # Also log locally, so I can actually plot these values later...
    with open(f"logs/{args.model_type}_variational_error.txt", "a+") as f:
        f.write(
                f"{args.tag}-{args.dataset}-{args.generalisation_task}-{args.num_datapoints}dp-{args.model_type}-"
                f"{args.optimiser}-{args.nonlinearity}-{early_stopping}-{num_epochs}-{args.lr_schedule}_schedule-"
                f"{device}, {str(args.hidden_units)}, {str(variational_error)}\n")

    if parse_bool(args.adjust_data_linearly):
        intercept, slope = da_train.linear_regressor.intercept_, da_train.linear_regressor.coef_[0]
        unadjusted_nn_preds = model_predictions.reshape(da_grid.x.shape) + linear(da_grid.x, intercept, slope)
        unadjusted_spline_preds = spline_predictions.reshape(da_grid.x.shape) + linear(da_grid.x, intercept, slope)
    else:
        unadjusted_nn_preds = model_predictions
        unadjusted_spline_preds = spline_predictions

    # Plot the predictions in the original, non-adjusted, non-normalised space.

    plot_data_vs_predictions(da_train.x, da_train.y, da_test.x, da_test.y,
                             unadjusted_nn_preds, grid,
                             unadjusted_spline_preds, fn_y, args, "original_space")

    # Plot the predictions in the adjusted, normalised space.
    plot_data_vs_predictions(x_tr, y_tr, x_te, y_te,
                             model_predictions, da_grid.x,
                             spline_predictions, da_grid.y, args, "adjusted_space")
