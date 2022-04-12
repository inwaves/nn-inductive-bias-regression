import numpy as np

from utils.maths import chebyshev_polynomial, normalise_data


def generate_sine_baseline():
    x_train = np.array([
        0,
        np.pi / 6,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        np.pi,
        4 * np.pi / 3,
        3 * np.pi / 2,
        5 * np.pi / 3,
        2 * np.pi
    ])
    x_test = np.array([])
    y_train, y_test = np.sin(x_train), np.array([])

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_sine_interpolation():
    x_train = np.array([
        0,
        np.pi / 6,
        np.pi / 3,
        np.pi / 2,
        3 * np.pi / 2,
        5 * np.pi / 3,
        2 * np.pi
    ])
    x_test = np.array([
        2 * np.pi / 3,
        np.pi,
        4 * np.pi / 3,
    ])
    y_train, y_test = np.sin(x_train), np.sin(x_test)
    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_sine_extrapolation():
    x_train = np.array([
        0,
        np.pi / 6,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        np.pi,
        4 * np.pi / 3,
    ])
    x_test = np.array([
        3 * np.pi / 2,
        5 * np.pi / 3,
        2 * np.pi
    ])
    y_train, y_test = np.sin(x_train), np.sin(x_test)
    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_square_baseline():
    """Generate data points for a square 'bump' (_---_)."""

    x_train = np.array([i for i in range(0, 10)])
    x_test = np.array([])

    y_train = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    y_test = np.array([])

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_square_interpolation():
    """Generate data points for a square 'bump' (_---_)."""

    x_train = np.array([0, 1, 2, 3, 7, 8, 9])
    x_test = np.array([4, 5, 6])

    y_train = np.array([0, 0, 0, 1, 0, 0, 0])
    y_test = np.array([1, 1, 0])

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_square_extrapolation():
    """Generate data points for a square 'bump' (_---_)."""

    x_train = np.array([0, 1, 2, 3, 4, 5, 6])
    x_test = np.array([7, 8, 9])

    y_train = np.array([0, 0, 0, 1, 1, 1, 0])
    y_test = np.array([0, 0, 0])

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_baseline():
    """Generate data points for a polynomial spline."""

    train_squared = np.array([-5, -4, -3, -2, -1])
    train_cubed = np.array([1, 23, 4, 5])

    x_train = np.concatenate((train_squared, train_cubed))
    x_test = np.array([])

    y_train = np.concatenate(train_squared ** 2, train_cubed ** 3 - train_cubed)
    y_test = np.array([])

    x_train, x_test = normalise_data(x_train, x_test)
    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_interpolation():
    """Generate data points for a polynomial spline."""

    train_squared = np.array([-5, -4, -3, -2])
    train_cubed = np.array([3, 4, 5])
    test_squared = np.array([-1])
    test_cubed = np.array([1, 2])

    x_train = np.concatenate((train_squared, train_cubed))
    x_test = np.concatenate((test_squared, test_cubed))
    y_train = np.concatenate(train_squared ** 2, train_cubed ** 3 - train_cubed)
    y_test = np.concatenate(test_squared ** 2, test_cubed ** 3 - test_cubed)

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_extrapolation():
    """Generate data points for a polynomial spline."""

    train_squared = np.array([-5, -4, -3, -2, -1])
    train_cubed = np.array([1, 2])
    x_test = np.array([3, 4, 5])

    x_train = np.concatenate((train_squared, train_cubed))
    y_train = np.concatenate(train_squared ** 2, train_cubed ** 3 - train_cubed)
    y_test = x_test ** 3 - x_test

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_chebyshev_baseline():
    """Generate data points for the 4th Chebyshev polynomial (16x^4 - 12x^2 + 1)."""

    x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    x_test = np.array([])

    y_train = chebyshev_polynomial(x_train, 4)
    y_test = np.array([])

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_chebyshev_interpolation():
    """Generate data points for the 4th Chebyshev polynomial (16x^4 - 12x^2 + 1)."""

    x_train = np.array([-5, -4, -3, -2, 2, 3, 4, 5])
    x_test = np.array([-1, 0, 1])

    y_train = chebyshev_polynomial(x_train, 4)
    y_test = chebyshev_polynomial(x_test, 4)

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_chebyshev_extrapolation():
    """Generate data points for the 4th Chebyshev polynomial (16x^4 - 12x^2 + 1)."""

    x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    x_test = np.array([3, 4, 5])

    y_train = chebyshev_polynomial(x_train, 4)
    y_test = chebyshev_polynomial(x_test, 4)

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_parabola_baseline():
    x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    x_test = np.array([])

    y_train = x_train ** 2
    y_test = x_test ** 2

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_parabola_interpolation():

    x_train = np.array([-5, -4, -3, -2, 2, 3, 4, 5])
    x_test = np.array([-1, 0, 1])

    y_train = x_train ** 2
    y_test = x_test ** 2

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def generate_parabola_extrapolation():
    x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    x_test = np.array([3, 4, 5])

    y_train = x_train ** 2
    y_test = x_test ** 2

    x_train, x_test = normalise_data(x_train, x_test)

    return x_train, y_train, x_test, y_test


def glue_dataset_portions(x_train, y_train, x_test, y_test):
    """This patches together the portions of the dataset such that we can
    apply the solution to the variational problem to all the data points."""
    tr = list(zip(x_train, y_train))
    te = list(zip(x_test, y_test))
    entire_set = np.array(sorted(tr + te))
    x_all = entire_set[:, 0]
    y_all = entire_set[:, 1]
    return x_all, y_all
