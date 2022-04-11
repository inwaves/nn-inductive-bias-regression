import argparse
import numpy as np
import torch
import wandb

from sklearn.linear_model import LinearRegression

from datasets.dataset import generate_deterministic_sine_interpolation, generate_sine_extrapolation_dataset, \
    generate_parabola, generate_square_interpolation_dataset, generate_polynomial_spline_interpolation_dataset, \
    generate_polynomial_spline_extrapolation_dataset, generate_chebyshev_polynomial_interpolation_dataset, \
    generate_deterministic_sine_baseline
from models.mlp import MLP
from models.shallow_relu import AsiShallowRelu, ShallowRelu, PlainTorchAsiShallowRelu
from utils.custom_dataloader import CustomDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parses command-line arguments corresponding to experiment parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_units", "-n", default=100, type=int, help="Number of hidden units (n).")
    parser.add_argument("--log_every_k_steps", "-l", default=100, type=int, help="Log the loss every k steps.")
    parser.add_argument("--adjust_data_linearly", "-a", default=False, type=bool, help="Adjust the data linearly?")
    parser.add_argument("--num_samples", "-s", default=7, type=int,
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


def select_dataset(args):
    # TODO: Add a unit test for this.
    """Select the dataset to use."""
    if args.dataset == "sine" and args.generalisation_task == "interpolation":
        return generate_deterministic_sine_interpolation()
    elif args.dataset == "sine" and args.generalisation_task == "extrapolation":
        return generate_sine_extrapolation_dataset(num_train_datapoints=args.num_samples)
    elif args.dataset == "sine" and args.generalisation_task == "baseline":
        return generate_deterministic_sine_baseline()
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
                       "num_samples": args.num_samples,
                       "adjust_data_linearly": args.adjust_data_linearly})

    # A little bit of annoying boilerplate that might solve my issue.
    x_train, y_train, x_test, y_test = select_dataset(args)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    print(f"Data inside the setup is on device: {x_train.device}")

    # Adjust the data linearly.
    if args.adjust_data_linearly:
        y_train = adjust_data_linearly(x_train, y_train)

    training_data = np.array(list(zip(x_train, y_train)))
    test_data = np.array(list(zip(x_test, y_test)))

    custom_dataloader = CustomDataLoader(training_data, test_data, device)
    train_dataloader = custom_dataloader.train_dataloader()
    test_dataloader = custom_dataloader.test_dataloader() if len(x_test) > 0 else None

    if args.model_type == "ASIShallowRelu":
        model = AsiShallowRelu(args.hidden_units,
                               input_dim=1,
                               output_dim=1,
                               lr=args.learning_rate).to(device).float()
    elif args.model_type == "ShallowRelu":
        model = ShallowRelu(args.hidden_units, 1, 1, lr=args.learning_rate).to(device).float()
    elif args.model_type == "PlainTorchAsiShallowRelu":
        model = PlainTorchAsiShallowRelu(args.hidden_units, 1, 1).to(device).float()
    print(f"Model is on device: {device}")

    return train_dataloader, test_dataloader, x_train, y_train, x_test, y_test, args, model
