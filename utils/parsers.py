import argparse

from torch import nn as nn


def parse_nonlinearity(nonlinearity_type):
    if nonlinearity_type == "relu":
        return nn.ReLU()
    elif nonlinearity_type == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif nonlinearity_type == "gelu":
        return nn.GELU()
    elif nonlinearity_type == "elu":
        return nn.ELU()
    elif nonlinearity_type == "sigmoid":
        return nn.Sigmoid()
    elif nonlinearity_type == "tanh":
        return nn.Tanh()


def parse_bool(arg):
    arg = arg.lower()
    if arg in ["true", "yes", "t", "1", "y"]:
        return True
    elif arg in ["false", "no", "f," "0", "n"]:
        return False
    else:
        raise ValueError("Argument must be a valid boolean value.")


def parse_args():
    """Parses command-line arguments corresponding to experiment parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_units", "-n", default=100, type=int, help="Number of hidden units (n).")
    parser.add_argument("--log_every_k_steps", "-l", default=100, type=int, help="Log the loss every k steps.")
    parser.add_argument("--adjust_data_linearly", "-a", default="True", type=str, help="Adjust the data linearly?")
    parser.add_argument("--normalise", default="True", type=str, help="Normalise the data?")
    parser.add_argument("--num_samples", "-s", default=7, type=int,
                        help="Number of points in the training dataset.")
    parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                        help="Learning rate of the optimiser.")
    parser.add_argument("--model_type", "-m", default="ASIShallowRelu", type=str, help="Select from ASIShallowRelu, "
                                                                                       "ShallowRelu, MLP.")
    parser.add_argument("--dataset", "-d", default="sine", type=str, help="Select from constant, linear, sine, "
                                                                          "parabola, chebyshev_polynomial, "
                                                                          "polynomial_spline.")
    parser.add_argument("--generalisation_task", "-g", default="interpolation", type=str, help="Select from baseline, "
                                                                                               "interpolation or "
                                                                                               "extrapolation.")
    parser.add_argument("--nonlinearity", "-nl", default="relu", type=str, help="Select from relu, leaky_relu, gelu, "
                                                                                "elu, sigmoid, tanh.")
    args = parser.parse_args()

    return args