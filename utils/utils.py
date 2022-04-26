import argparse
import torch
import wandb

from sklearn.linear_model import LinearRegression
from functools import partial
from datasets.dataset import *
from models.mlp import MLP
from models.shallow_relu import AsiShallowRelu, ShallowRelu, PlainTorchAsiShallowRelu
from utils.custom_dataloader import CustomDataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """Parses command-line arguments corresponding to experiment parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_units", "-n", default=100, type=int, help="Number of hidden units (n).")
    parser.add_argument("--log_every_k_steps", "-l", default=100, type=int, help="Log the loss every k steps.")
    parser.add_argument("--adjust_data_linearly", "-a", default=True, type=bool, help="Adjust the data linearly?")
    parser.add_argument("--num_samples", "-s", default=7, type=int,
                        help="Number of points in the training dataset.")
    parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate of the optimiser.")
    parser.add_argument("--model_type", "-m", default="ASIShallowRelu", type=str, help="Select from ASIShallowRelu, "
                                                                                       "ShallowRelu, MLP.")
    parser.add_argument("--dataset", "-d", default="sine", type=str, help="Select from constant, linear, sine, "
                                                                          "parabola, chebyshev_polynomial, "
                                                                          "polynomial_spline.")
    parser.add_argument("--generalisation_task", "-g", default="interpolation", type=str, help="Select from baseline, "
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

    return np.sqrt(np.mean((network_predictions.reshape(variational_predictions.shape) - variational_predictions) ** 2))


def select_dataset(args):
    # TODO: Add a unit test for this.
    """Select the dataset to use."""
    if args.dataset == "linear" and args.generalisation_task == "baseline":
        return generate_linear_baseline(), linear
    elif args.dataset == "linear" and args.generalisation_task == "interpolation":
        return generate_linear_interpolation(), linear
    if args.dataset == "linear" and args.generalisation_task == "extrapolation":
        return generate_linear_extrapolation(), linear
    elif args.dataset == "constant" and args.generalisation_task == "baseline":
        return generate_constant_baseline(), constant
    elif args.dataset == "constant" and args.generalisation_task == "interpolation":
        return generate_constant_interpolation(), constant
    elif args.dataset == "constant" and args.generalisation_task == "extrapolation":
        return generate_constant_extrapolation(), constant
    elif args.dataset == "sine" and args.generalisation_task == "baseline":
        return generate_sine_baseline(), sin
    elif args.dataset == "sine" and args.generalisation_task == "interpolation":
        return generate_sine_interpolation(), sin
    elif args.dataset == "sine" and args.generalisation_task == "extrapolation":
        return generate_sine_extrapolation(), sin
    elif args.dataset == "parabola" and args.generalisation_task == "baseline":
        return generate_parabola_baseline(), parabola
    elif args.dataset == "parabola" and args.generalisation_task == "interpolation":
        return generate_parabola_interpolation(), parabola
    elif args.dataset == "parabola" and args.generalisation_task == "extrapolation":
        return generate_parabola_extrapolation(), parabola
    elif args.dataset == "square" and args.generalisation_task == "baseline":
        return generate_square_baseline(), square
    elif args.dataset == "square" and args.generalisation_task == "interpolation":
        return generate_square_interpolation(), square
    elif args.dataset == "square" and args.generalisation_task == "extrapolation":
        return generate_square_extrapolation(), square
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "baseline":
        return generate_polynomial_spline_baseline(), polynomial_spline
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "interpolation":
        return generate_polynomial_spline_interpolation(), polynomial_spline
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "extrapolation":
        return generate_polynomial_spline_extrapolation(), polynomial_spline
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "baseline":
        return generate_chebyshev_baseline(), partial(chebyshev_polynomial, n=4)
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "interpolation":
        return generate_chebyshev_interpolation(), partial(chebyshev_polynomial, n=4)
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "extrapolation":
        return generate_chebyshev_extrapolation(), partial(chebyshev_polynomial, n=4)


def setup():
    args = parse_args()
    wandb.init(project="generalisation",
               entity="inwaves",
               config={"model_type": args.model_type,
                       "hidden_units": args.hidden_units,
                       "lr": args.learning_rate,
                       "dataset": args.dataset,
                       "generalisation_task": args.generalisation_task,
                       "adjust_data_linearly": args.adjust_data_linearly})

    # Set up the data.
    raw_x_train, raw_y_train, raw_x_test, raw_y_test, fn = select_dataset(args)
    x_train, x_test = normalise_data(raw_x_train, raw_x_test)

    # Adjust the data linearly.
    if args.adjust_data_linearly:
        y_train = adjust_data_linearly(x_train, raw_y_train)
        y_test = adjust_data_linearly(x_test, raw_y_test)

    training_data = np.array(list(zip(x_train, y_train)))
    test_data = np.array(list(zip(x_test, y_test)))

    custom_dataloader = CustomDataLoader(training_data, test_data, device)
    train_dataloader = custom_dataloader.train_dataloader()
    test_dataloader = custom_dataloader.test_dataloader() if len(raw_x_test) > 0 else None

    # Set up the model.
    if args.model_type == "ASIShallowRelu":
        model = AsiShallowRelu(args.hidden_units, 1, 1, lr=args.learning_rate).to(device).float()
    elif args.model_type == "ShallowRelu":
        model = ShallowRelu(args.hidden_units, 1, 1, lr=args.learning_rate).to(device).float()
    elif args.model_type == "PlainTorchAsiShallowRelu":
        model = PlainTorchAsiShallowRelu(args.hidden_units, 1, 1).to(device).float()
    elif args.model_type == "MLP":
        model = MLP(args.hidden_units, 1, 1, lr=args.learning_rate).to(device).float()

    return train_dataloader, test_dataloader, (x_train, y_train, x_test, y_test), (raw_x_train, raw_y_train, raw_x_test, raw_y_test), args, model, fn