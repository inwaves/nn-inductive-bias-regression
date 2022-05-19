import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scipy.interpolate import CubicSpline
from datasets.dataset import glue_dataset_portions

from utils.data_adjuster import DataAdjuster
from utils.maths import linear, mean_squared_error
from utils.utils import parse_bool, setup
from utils.plotting import plot_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Make deterministic for reproducibility, or comment out to average runs.
# pl.seed_everything(1337)


if __name__ == '__main__':
    train_dataloader, test_dataloader, da_train, da_test, args, model, fn = setup()

    early_stopping_callback = EarlyStopping(monitor="train_loss", min_delta=1e-8, patience=3)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpointing_callback = ModelCheckpoint(dirpath="ckpts",
                                             every_n_epochs=args.val_frequency)

    wandb_logger = WandbLogger(project="generalisation")
    print(wandb.run.name)

    max_epochs = args.num_epochs
    if parse_bool(args.early_stopping):
        # This control flow is needed to be able to run this script
        # on either CPU (locally) or GPU (on a cluster).

        # TODO: Fix this, DRY
        if device == "cuda":
            trainer = pl.Trainer(max_epochs=-1,
                                 callbacks=[early_stopping_callback, lr_monitor_callback],
                                 accelerator="gpu",
                                 devices=1,
                                 logger=wandb_logger,
                                 log_every_n_steps=args.log_every_k_steps,
                                 check_val_every_n_epoch=args.val_frequency)
        else:
            trainer = pl.Trainer(max_epochs=-1,
                                 callbacks=[early_stopping_callback, lr_monitor_callback, checkpointing_callback],
                                 accelerator="cpu",
                                 logger=wandb_logger,
                                 log_every_n_steps=args.log_every_k_steps,
                                 check_val_every_n_epoch=args.val_frequency)
    else:
        if device == "cuda":
            trainer = pl.Trainer(max_epochs=max_epochs,
                                 callbacks=[lr_monitor_callback],
                                 accelerator="gpu",
                                 devices=1,
                                 logger=wandb_logger,
                                 log_every_n_steps=args.log_every_k_steps,
                                 check_val_every_n_epoch=args.val_frequency)
        else:
            trainer = pl.Trainer(max_epochs=max_epochs,
                                 callbacks=[lr_monitor_callback],
                                 accelerator="cpu",
                                 logger=wandb_logger,
                                 log_every_n_steps=args.log_every_k_steps,
                                 check_val_every_n_epoch=args.val_frequency)

    # Model is fit to the normalised, linearly adjusted data.
    tic = time.time()
    trainer.fit(model,
                train_dataloader=train_dataloader,
                val_dataloaders=[train_dataloader],
                )

    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    trainer.test(model=model, dataloaders=[test_dataloader])

    model = model.to(device)

    # Using adjusted data...
    # ...fit the cubic spline.
    spline = CubicSpline(da_train.x, da_train.y)

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

    # Calculate the difference between g* and the NN function on the grid.
    spline_predictions = spline(da_grid.x)
    model_predictions = model(torch.tensor(da_grid.x).float().unsqueeze(1)).cpu().detach().numpy()
    variational_error = mean_squared_error(spline_predictions, model_predictions)

    # Log locally, so I can actually plot these values later...
    with open("logs/nn_vs_variational_solution_error.txt", "a") as f:
        early_stopping = "earlystopping" if parse_bool(args.early_stopping) else "no_earlystopping"
        n_epochs = f"{max_epochs}epochs"
        lrs = f"{args.lr_schedule}_schedule"
        f.write(f"{args.dataset}-{args.generalisation_task}-{args.num_datapoints}dp-{args.model_type}-{args.optimiser}-"
                f"{args.nonlinearity}-{early_stopping}-{n_epochs}-{lrs}-{device}, {str(args.hidden_units)}, {str(variational_error)}\n")

    wandb.summary["nn_vs_solution_error"] = variational_error

    if parse_bool(args.adjust_data_linearly):
        intercept, slope = da_train.linear_regressor.intercept_, da_train.linear_regressor.coef_[0]
        unadjusted_nn_preds = model_predictions.reshape(da_grid.x.shape) + linear(da_grid.x, intercept, slope)
        unadjusted_spline_preds = spline_predictions.reshape(da_grid.x.shape) + linear(da_grid.x, intercept, slope)
    else:
        unadjusted_nn_preds = model_predictions
        unadjusted_spline_preds = spline_predictions

    # Plot the predictions in the original, non-adjusted, non-normalised space.

    plot = plot_data_vs_predictions(da_train.x, da_train.y, da_test.x, da_test.y,
                                    unadjusted_nn_preds, grid,
                                    unadjusted_spline_preds, fn_y, args)

    # Wrap up any hanging logger.
    wandb.finish()
