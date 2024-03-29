import argparse
from functools import partial

from torch import nn as nn
import torch.nn.functional as F
import torch


def parse_schedule(scheduler, optimiser):
    scheduler = scheduler.lower()
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100)
    elif scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                          mode="min",
                                                          factor=0.5,
                                                          patience=100,
                                                          threshold=1e-4,
                                                          threshold_mode="abs",
                                                          verbose=False)
    return None


def parse_optimiser(optimiser):
    optimiser = optimiser.lower()
    if optimiser == "sgd":
        return torch.optim.SGD
    elif optimiser == "adam":
        return torch.optim.Adam
    elif optimiser == "momentum":
        return partial(torch.optim.SGD, momentum=0.9)
    print("Optimiser parsing error, optimiser is none.")
    return None


def parse_loss_fn(loss):
    loss = loss.lower()
    if loss == "mse":
        return F.mse_loss
    elif loss == "mae":
        return F.l1_loss
    elif loss == "huber":
        return F.huber_loss
    print("Loss function parsing error, loss is none.")
    return None


def parse_nonlinearity(nonlinearity):
    nonlinearity = nonlinearity.lower()
    if nonlinearity == "relu":
        return nn.ReLU()
    elif nonlinearity == "leaky_relu":
        return nn.LeakyReLU(0.2)
    elif nonlinearity == "gelu":
        return nn.GELU()
    elif nonlinearity == "elu":
        return nn.ELU()
    elif nonlinearity == "sigmoid":
        return nn.Sigmoid()
    elif nonlinearity == "tanh":
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
    parser.add_argument("--a_b", "-ab", default=2, type=int, help="Network biases are sampled from U(-a_b, a_b)")
    parser.add_argument("--adjust_data_linearly", "-a", default="True", type=str, help="Adjust the data linearly?")
    parser.add_argument("--a_w", "-aw", default=1, type=int, help="Network weights are sampled from U(-a_w, a_w)")
    parser.add_argument("--dataset", "-d", default="sine", type=str, help="Select from constant, linear, sine, "
                                                                          "parabola, chebyshev_polynomial, "
                                                                          "polynomial_spline, random, square.")
    parser.add_argument("--early_stopping", "-es", default="True", type=str, help="Use early stopping?")
    parser.add_argument("--generalisation_task", "-g", default="interpolation", type=str, help="Select from baseline, "
                                                                                               "interpolation or "
                                                                                               "extrapolation.")
    parser.add_argument("--grid_resolution", "-gr", default=100, type=int, help="How many data points in the grid?")
    parser.add_argument("--hidden_units", "-n", default=100, type=int, help="Number of hidden units (n).")
    parser.add_argument("--init", "-i", default="uniform", type=str, help="Uniform or normal initialisation of "
                                                                          "params.")
    parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                        help="Learning rate of the optimiser.")
    parser.add_argument("--lr_schedule", "-sc", default="none", type=str, help="Select from cosine, plateau or none.")
    parser.add_argument("--loss", "-lo", default="mse", type=str, help="Select from MSE, MAE, Huber.")
    parser.add_argument("--log_every_n_steps", "-l", default=500, type=int, help="Log the loss every k steps.")
    parser.add_argument("--model_type", "-m", default="ASIShallowRelu", type=str, help="Select from ASIShallowRelu, "
                                                                                       "ShallowRelu, MLP.")
    parser.add_argument("--model_checkpoint", "-mc", default="False", type=str, help="Should checkpoint model?")
    parser.add_argument("--normalise", default="True", type=str, help="Normalise the data?")
    parser.add_argument("--nonlinearity", "-nl", default="relu", type=str, help="Select from relu, leaky_relu, gelu, "
                                                                                "elu, sigmoid, tanh.")
    parser.add_argument("--num_datapoints", "-nd", default=10, type=int,
                        help="Number of points in the (training+test) datasets. The ratio is always 70:30.")
    parser.add_argument("--num_epochs", "-ne", default=10000, type=int, help="Number of epochs to run for.")
    parser.add_argument("--optimiser", "-o", default="sgd", type=str, help="Select from SGD, Adam, momentum")
    parser.add_argument("--tag", "-t", default="untagged", type=str, help="Add a tag for this experiment.")
    parser.add_argument("--val_frequency", "-va", default=1000, type=int, help="How often to compute validation error.")
    args = parser.parse_args()

    return args
