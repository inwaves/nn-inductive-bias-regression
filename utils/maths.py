import numpy as np


def chebyshev_polynomial(x, n):
    """Calculate the nth Chebyshev polynomial of x."""
    if x.size == 0:
        return x
    if n == 0:
        return np.array([1])
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * chebyshev_polynomial(x, n - 1) - chebyshev_polynomial(x, n - 2)


def constant(x):
    return np.ones(x.shape)


def linear(x, intercept=3, slope=2):
    return slope * x + intercept


def sin(x):
    return np.sin(x)


def square(x):
    return 1 if 3 <= x < 6 else 0


def polynomial_spline(x):
    return x ** 2 if x < 0 else x ** 3 - x


def parabola(x):
    return x ** 2


def normalise_data(x, minval=None, maxval=None):
    """Normalise training data to lie between -1 and 1. Normalise test data to lie between minval and maxval"""

    minval = np.min(x) if minval is None else minval
    maxval = np.max(x) if maxval is None else maxval
    x = ((2 * (x - minval)) / (maxval - minval)) - 1

    return x, minval, maxval


def mean_squared_error(targets, model_preds):
    """Calculate the mean squared error between a target variable and model predictions."""

    return np.sqrt(np.mean((model_preds.reshape(targets.shape) - targets) ** 2))
