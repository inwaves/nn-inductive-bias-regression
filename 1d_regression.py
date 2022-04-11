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
from utils.plotting import plot_sin_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"
pl.seed_everything(1337)


if __name__ == '__main__':
    train_dataloader, test_dataloader, x_train, y_train, x_test, y_test, args, model = setup()

    early_stopping_callback = EarlyStopping(monitor="train_loss", min_delta=1e-8, patience=3)
    wandb_logger = WandbLogger(project="generalisation")
    trainer = pl.Trainer(max_epochs=-1,
                         callbacks=[early_stopping_callback],
                         gpus=1,
                         logger=wandb_logger,
                         log_every_n_steps=args.log_every_k_steps, )
    tic = time.time()
    trainer.fit(model, train_dataloader)
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    # trainer.test(model=model, dataloaders=[test_dataloader])

    x_all, y_all = glue_dataset_portions(x_train, y_train, x_test, y_test)

    # Grid is used to plot g* correctly, otherwise it doesn't match the actual function
    # because there are not that many data points in x_all.
    grid = np.linspace(np.min(x_all), np.max(x_all), 100)

    # Fit the cubic spline to the training data only.
    spline = CubicSpline(x_train, y_train)

    # Find NN predictions for all data points (train + test).
    y_pred = model(torch.tensor(x_all).float().unsqueeze(1).to(device)).detach().numpy()

    # Calculate the difference between the NN function and g* on the training data.
    error = variational_solution_vs_neural_network(spline(x_train), model(torch.tensor(x_train)
                                                                          .float()
                                                                          .unsqueeze(1)
                                                                          .to(device))
                                                   .detach().numpy())

    # Log locally, so I can actually plot these values later...
    with open("logs/nn_vs_variational_solution_error.txt", "a") as f:
        f.write(f"{str(args.hidden_units)}, {str(error)}\n")

    wandb.log({"nn_vs_solution_error": error})
    plot_sin_data_vs_predictions(x_train, y_train, x_test, y_test, y_pred, x_all, grid, spline(grid))
