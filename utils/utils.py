import torch
import wandb

from sklearn.linear_model import LinearRegression
from functools import partial

from datasets.dataset import *
from models.mlp import MLP
from models.shallow_relu import AsiShallowNetwork, ShallowNetwork, PlainTorchAsiShallowRelu
from utils.custom_dataloader import CustomDataLoader
from utils.parsers import parse_args, parse_bool
from utils.data_adjuster import DataAdjuster

device = "cuda" if torch.cuda.is_available() else "cpu"


def select_model(model_type, hidden_units, learning_rate):
    if model_type == "asishallowrelu":
        model = AsiShallowNetwork(hidden_units, 1, 1, lr=learning_rate).to(device).float()
    elif model_type == "shallowrelu":
        model = ShallowNetwork(hidden_units, 1, 1, lr=learning_rate).to(device).float()
    elif model_type == "plaintorchasishallowrelu":
        model = PlainTorchAsiShallowRelu(hidden_units, 1, 1).to(device).float()
    elif model_type == "mlp":
        model = MLP(hidden_units, 1, 1, lr=learning_rate).to(device).float()
    else:
        print(f"Error: model type {model_type} not supported.")
        model = None
    return model


def adjust_data_linearly(x_train, y_train):
    """Fit a linear regression model."""

    linear_regressor = LinearRegression().fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    linreg_pred = linear_regressor.predict(x_train.reshape(-1, 1)).reshape(-1)
    residual = y_train - linreg_pred

    return residual, linear_regressor


def calculate_spline_vs_model_error(variational_predictions, network_predictions):
    """Calculate the mean square error between the two predictions."""

    return np.sqrt(np.mean((network_predictions.reshape(variational_predictions.shape) - variational_predictions) ** 2))


def select_dataset(args):
    # TODO: Add a unit test for this.
    """Select the dataset to use."""
    if args.dataset == "linear" and args.generalisation_task == "baseline":
        return generate_linear_dataset(), linear
    elif args.dataset == "linear" and args.generalisation_task == "interpolation":
        return generate_linear_interpolation(), linear
    if args.dataset == "linear" and args.generalisation_task == "extrapolation":
        return generate_linear_extrapolation(), linear
    elif args.dataset == "constant" and args.generalisation_task == "baseline":
        return generate_constant_dataset(), constant
    elif args.dataset == "constant" and args.generalisation_task == "interpolation":
        return generate_constant_interpolation(), constant
    elif args.dataset == "constant" and args.generalisation_task == "extrapolation":
        return generate_constant_extrapolation(), constant
    elif args.dataset == "sine" and args.generalisation_task == "baseline":
        return generate_sine_dataset(), sin
    elif args.dataset == "sine" and args.generalisation_task == "interpolation":
        return generate_sine_interpolation(), sin
    elif args.dataset == "sine" and args.generalisation_task == "extrapolation":
        return generate_sine_extrapolation(), sin
    elif args.dataset == "parabola" and args.generalisation_task == "baseline":
        return generate_parabola_dataset(), parabola
    elif args.dataset == "parabola" and args.generalisation_task == "interpolation":
        return generate_parabola_interpolation(), parabola
    elif args.dataset == "parabola" and args.generalisation_task == "extrapolation":
        return generate_parabola_extrapolation(), parabola
    elif args.dataset == "square" and args.generalisation_task == "baseline":
        return generate_square_dataset(), square
    elif args.dataset == "square" and args.generalisation_task == "interpolation":
        return generate_square_interpolation(), square
    elif args.dataset == "square" and args.generalisation_task == "extrapolation":
        return generate_square_extrapolation(), square
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "baseline":
        return generate_polynomial_spline_dataset(), polynomial_spline
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "interpolation":
        return generate_polynomial_spline_interpolation(), polynomial_spline
    elif args.dataset == "polynomial_spline" and args.generalisation_task == "extrapolation":
        return generate_polynomial_spline_extrapolation(), polynomial_spline
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "baseline":
        return generate_chebyshev_dataset(), partial(chebyshev_polynomial, n=4)
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "interpolation":
        return generate_chebyshev_interpolation(), partial(chebyshev_polynomial, n=4)
    elif args.dataset == "chebyshev_polynomial" and args.generalisation_task == "extrapolation":
        return generate_chebyshev_extrapolation(), partial(chebyshev_polynomial, n=4)


def setup():
    args = parse_args()
    wandb.init(project="generalisation",
               entity="inwaves",
               config={"model_type":           args.model_type,
                       "nonlinearity":         args.nonlinearity,
                       "hidden_units":         args.hidden_units,
                       "lr":                   args.learning_rate,
                       "dataset":              args.dataset,
                       "generalisation_task":  args.generalisation_task,
                       "adjust_data_linearly": args.adjust_data_linearly,
                       "normalise":            args.normalise})

    # Set up the data.
    (x_train, y_train, x_test, y_test), fn = select_dataset(args)
    da_train = DataAdjuster(x_train, y_train)
    da_test = DataAdjuster(x_test, y_test, da_train.x_min, da_train.x_max)

    if parse_bool(args.normalise):
        da_train.normalise()
        da_test.normalise()
        print(f"Normalising data because flag is: {args.normalise}")

    # Adjust the data linearly.
    if parse_bool(args.adjust_data_linearly):
        da_train.adjust()
        da_test.adjust()
        print(f"Adjusting data linearly because flag is: {args.adjust_data_linearly}")

    training_data = np.array(list(zip(da_train.x, da_train.y)))
    test_data = np.array(list(zip(da_test.x, da_test.y)))

    custom_dataloader = CustomDataLoader(training_data, test_data, device)
    train_dataloader = custom_dataloader.train_dataloader()
    test_dataloader = custom_dataloader.test_dataloader() if len(da_test.x) > 0 else None

    # Set up the model.
    model = select_model(args.model_type.lower(), args.hidden_units, args.learning_rate)

    return train_dataloader, test_dataloader, da_train, da_test, args, model, fn
