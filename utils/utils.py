import torch
import wandb

from sklearn.linear_model import LinearRegression
from functools import partial

from datasets.dataset import *
from models.mlp import MLP
from models.shallow_relu import AsiShallowNetwork, ShallowNetwork, PlainTorchAsiShallowRelu
from utils.custom_dataloader import CustomDataLoader
from utils.parsers import parse_args, parse_bool, parse_optimiser, parse_schedule
from utils.data_adjuster import DataAdjuster

device = "cuda" if torch.cuda.is_available() else "cpu"


def select_model(model_type, hidden_units, learning_rate, optimiser, schedule):
    optimiser = parse_optimiser(optimiser)
    model_type = model_type.lower()
    if model_type == "asishallowrelu":
        model = AsiShallowNetwork(hidden_units, 1, 1, lr=learning_rate, optimiser=optimiser, schedule=schedule).to(device).float()
    elif model_type == "shallowrelu":
        model = ShallowNetwork(hidden_units, 1, 1, lr=learning_rate, optimiser=optimiser, schedule=schedule).to(device).float()
    elif model_type == "plaintorchasishallowrelu":
        model = PlainTorchAsiShallowRelu(hidden_units, 1, 1, "relu").to(device).float()
    elif model_type == "mlp":
        model = MLP(hidden_units, 1, 1, lr=learning_rate, optimiser=optimiser, schedule=schedule).to(device).float()
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
    if args.dataset == "linear":
        return generate_linear_dataset(args.generalisation_task, args.num_datapoints), linear
    elif args.dataset == "constant":
        return generate_constant_dataset(args.generalisation_task, args.num_datapoints), constant
    elif args.dataset == "sine":
        return generate_sine_dataset(args.generalisation_task, args.num_datapoints), sin
    elif args.dataset == "parabola":
        return generate_parabola_dataset(args.generalisation_task, args.num_datapoints), parabola
    elif args.dataset == "square":
        return generate_square_dataset(args.generalisation_task, args.num_datapoints), square
    elif args.dataset == "polynomial_spline":
        return generate_polynomial_spline_dataset(args.generalisation_task, args.num_datapoints), polynomial_spline
    elif args.dataset == "chebyshev_polynomial":
        return generate_chebyshev_dataset(args.generalisation_task, args.num_datapoints), partial(chebyshev_polynomial,
                                                                                                  n=4)
    # This isn't fully built out yet, commenting out, so it doesn't break the code.
    # elif args.dataset == "random":
    #     return generate_random_dataset(args.generalisation_task, args.num_datapoints), random


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
                       "normalise":            args.normalise,
                       "num_datapoints":       args.num_datapoints,
                       "optimiser":            args.optimiser,
                       "lr_schedule":          args.lr_schedule, })

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
    model = select_model(args.model_type, args.hidden_units, args.learning_rate, args.optimiser, args.lr_schedule)

    return train_dataloader, test_dataloader, da_train, da_test, args, model, fn
