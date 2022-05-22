from functools import partial

import torch

from datasets.dataset import generate_chebyshev_dataset, generate_constant_dataset, generate_linear_dataset, \
    generate_parabola_dataset, \
    generate_polynomial_spline_dataset, generate_sine_dataset, generate_square_dataset
from models.mlp import MLP
from models.shallow_relu import AsiShallowNetwork, PlainTorchAsiShallowRelu, ShallowNetwork
from utils.maths import chebyshev_polynomial, constant, linear, parabola, polynomial_spline, sin, square
from utils.parsers import parse_optimiser

device = "cuda" if torch.cuda.is_available() else "cpu"


def select_model(da_train, da_test, fn, adjust_data_linearly, normalise, grid_resolution, model_type, hidden_units,
                 learning_rate, optimiser, schedule, init, a_w, a_b):
    optimiser = parse_optimiser(optimiser)
    model_type = model_type.lower()
    if model_type == "asishallowrelu":
        model = AsiShallowNetwork(da_train=da_train,
                                  da_test=da_test,
                                  fn=fn,
                                  adjust_data_linearly=adjust_data_linearly,
                                  normalise=normalise,
                                  grid_resolution=grid_resolution,
                                  hidden_units=hidden_units,
                                  input_dim=1,
                                  output_dim=1,
                                  lr=learning_rate,
                                  optimiser=optimiser,
                                  schedule=schedule,
                                  init=init,
                                  a_w=a_w,
                                  a_b=a_b
                                  ).to(device).float()
    elif model_type == "shallowrelu":
        model = ShallowNetwork(da_train=da_train,
                                  da_test=da_test,
                                  fn=fn,
                                  adjust_data_linearly=adjust_data_linearly,
                                  normalise=normalise,
                                  grid_resolution=grid_resolution,
                                  hidden_units=hidden_units,
                                  input_dim=1,
                                  output_dim=1,
                                  lr=learning_rate,
                                  optimiser=optimiser,
                                  schedule=schedule,
                                  init=init
                                  ).to(device).float()
    elif model_type == "plaintorchasishallowrelu":
        model = ShallowNetwork(da_train=da_train,
                               da_test=da_test,
                               fn=fn,
                               adjust_data_linearly=adjust_data_linearly,
                               normalise=normalise,
                               grid_resolution=grid_resolution,
                               hidden_units=hidden_units,
                               input_dim=1,
                               output_dim=1,
                               lr=learning_rate,
                               optimiser=optimiser,
                               schedule=schedule,
                               init=init
                               ).to(device).float()
        model = PlainTorchAsiShallowRelu(n=hidden_units,
                                         input_dim=1,
                                         output_dim=1,
                                         nonlinearity="relu").to(device).float()
    elif model_type == "mlp":
        model = MLP(da_train=da_train,
                    da_test=da_test,
                    fn=fn,
                    adjust_data_linearly=adjust_data_linearly,
                    normalise=normalise,
                    grid_resolution=grid_resolution,
                    hidden_units=hidden_units,
                    input_dim=1,
                    output_dim=1,
                    lr=learning_rate,
                    optimiser=optimiser,
                    schedule=schedule,
                    init=init
                    ).to(device).float()
    else:
        print(f"Error: model type {model_type} not supported.")
        model = None
    return model


def select_dataset(args):
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
