import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from datasets.dataset import generate_sine_interpolation_dataset, generate_sine_extrapolation_dataset, \
    generate_parabola, generate_square_interpolation_dataset, \
    generate_polynomial_spline_interpolation_dataset, generate_polynomial_spline_extrapolation_dataset, \
    generate_chebyshev_polynomial_interpolation_dataset, glue_dataset_portions
from models.mlp import MLP
from models.shallow_relu import AsiShallowRelu, ShallowRelu
from utils.plotting import plot_sin_data_vs_predictions

# Initialisation.
device = "cuda" if torch.cuda.is_available() else "cpu"
pl.seed_everything(1337)


# TODO: make these into CLI arguments.

def parse_args():
    """Parses command-line arguments corresponding to experiment parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_units", "-n", default=100, type=int, help="Number of hidden units (n).")
    parser.add_argument("--num_samples", "-s", default=50, type=int,
                        help="Number of points in the training dataset.")
    parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate of the optimiser.")
    parser.add_argument("--model_type", "-m", default="ASIShallowRelu", type=str, help="Select from ASIShallowRelu, "
                                                                                       "ShallowRelu, MLP.")
    parser.add_argument("--dataset", "-d", default="sine", type=str, help="Select from sine, parabola, chebyshev, "
                                                                          "spline.")
    parser.add_argument("--generalisation_task", "-g", default="interpolation", type=str, help="Select from "
                                                                                               "interpolation or "
                                                                                               "extrapolation.")
    args = parser.parse_args()

    return args


def adjust_data_linearly(x_train, y_train):
    """Fit a linear regression model."""

    linear_regressor = LinearRegression().fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    residual = y_train - linear_regressor.predict(x_train.reshape(-1, 1)).reshape(-1)

    return residual


def variational_solution_vs_neural_network(variational_predictions, network_predictions):
    """Calculate the mean square error between the two predictions."""

    return np.mean((network_predictions - variational_predictions) ** 2)


# TODO: Add a unit test for this.
def select_dataset(args):
    """Select the dataset to use."""
    if args.dataset == "sine" and args.generalisation_task == "interpolation":
        return generate_sine_interpolation_dataset(gap_size=np.pi / 2, num_train_datapoints=args.num_samples)
    elif args.dataset == "sine" and args.generalisation_task == "extrapolation":
        return generate_sine_extrapolation_dataset(num_train_datapoints=args.num_samples)
    elif args.dataset == "parabola" and args.generalisation_task == "interpolation":
        return generate_parabola(num_train_datapoints=args.num_samples)
    elif args.dataset == "square" and args.generalisation_task == "interpolation":
        return generate_square_interpolation_dataset(gap_size=1,
                                                     num_train_datapoints=args.num_samples)
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "interpolation":
        return generate_polynomial_spline_interpolation_dataset(gap_size=-1, num_train_datapoints=args.num_samples)
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "extrapolation":
        return generate_polynomial_spline_extrapolation_dataset(num_train_datapoints=args.num_samples)
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "interpolation":
        return generate_chebyshev_polynomial_interpolation_dataset(gap_size=-0.6,
                                                                   num_train_datapoints=args.num_samples)


def setup():
    args = parse_args()
    wandb.init(project="generalisation",
               entity="inwaves",
               config={"model_type": args.model_type,
                       "hidden_units": args.hidden_units,
                       "lr": args.learning_rate,
                       "dataset": args.dataset,
                       "generalisation_task": args.generalisation_task,
                       "num_samples": args.num_samples, })

    x_train, y_train, x_test, y_test = select_dataset(args)

    # Adjust the data linearly.
    y_train = adjust_data_linearly(x_train, y_train)
    training_data = np.array(list(zip(x_train, y_train)))
    test_data = np.array(list(zip(x_test, y_test)))

    # We're doing full-batch gradient descent, so the batch_size = n
    train_dataloader = DataLoader(training_data, batch_size=len(x_train))
    test_dataloader = DataLoader(test_data, batch_size=len(x_test))

    if args.model_type == "ASIShallowRelu":
        model = AsiShallowRelu(args.hidden_units,
                               input_dim=1,
                               output_dim=1,
                               lr=args.learning_rate).to(device).float()
    elif args.model_type == "ShallowRelu":
        model = ShallowRelu(args.hidden_units, 1, 1, lr=args.learning_rate).to(device).float()
    else:
        model = MLP(args.hidden_units, 1, 1, lr=args.lr).to(device).float()

    return train_dataloader, test_dataloader, x_train, y_train, x_test, y_test, args, model


if __name__ == '__main__':
    train_dataloader, test_dataloader, x_train, y_train, x_test, y_test, args, model = setup()

    early_stopping_callback = EarlyStopping(monitor="train_loss", min_delta=1e-8, patience=3)
    trainer = pl.Trainer(max_epochs=-1,
                         callbacks=[early_stopping_callback],
                         log_every_n_steps=1, )
    trainer.fit(model, train_dataloader)

    trainer.test(model=model, dataloaders=[test_dataloader])
    print(x_train, x_test)

    # Apply g* to all the data points.
    x_all, y_all = glue_dataset_portions(x_train, y_train, x_test, y_test)
    grid = np.linspace(np.min(x_all), np.max(x_all), 100)

    # Fit the cubic spline to the training data only.
    spline = CubicSpline(x_train, y_train)

    y_pred = model(torch.tensor(x_all).float().unsqueeze(1).to(device)).detach().numpy()

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
