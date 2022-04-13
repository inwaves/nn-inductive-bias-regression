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
    return 1


def linear(x):
    return 2 * x + 3


def sin(x):
    return np.sin(x)


def square(x):
    return 1 if 3 <= x < 6 else 0


def polynomial_spline(x):
    return x ** 2 if x < 0 else x ** 3 - x


def parabola(x):
    return x ** 2


def normalise_data(x_train, x_test):
    """Normalise the training and test data to lie between 0 and 1."""

    train_max = np.max(x_train)
    x_train = x_train / train_max
    x_test = x_test / train_max

    return x_train, x_test
