import numpy as np

from utils.maths import chebyshev_polynomial


# Scale x to be between -1, 1
def generate_sine_interpolation_dataset(gap_size, num_train_datapoints=10, num_test_datapoints=None):
    """Generate data points for a sine wave where the
    training set ranges [-2π, gap_size) u [2π-gap_size, 4π),
    and the test set ranges [-2π, 4π).
        :param gap_size: gap size between training and test set. gap size of π means no gap.
        :param num_train_datapoints: number of training data points to generate.
        :param num_test_datapoints: number of test data points to generate.
    """

    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    x_train = np.concatenate((np.linspace(-2 * np.pi, gap_size, num_train_datapoints // 2, endpoint=False),
                              np.linspace(2 * np.pi - gap_size, 4 * np.pi, num_train_datapoints // 2, endpoint=False)))
    x_test = np.linspace(gap_size, 2 * np.pi - gap_size, num_test_datapoints, endpoint=False)
    y_train = np.sin(x_train)
    y_test = np.sin(x_test)
    train_max = np.max(x_train)
    x_train /= train_max
    x_test /= train_max

    return x_train, y_train, x_test, y_test


def generate_sine_extrapolation_dataset(num_train_datapoints, num_test_datapoints=None):
    """Generate data points for extrapolating a sine function.
        :param num_train_datapoints: number of training data points to generate in interval [-2π, 2π).
        :param num_test_datapoints: number of test data points to generate in [2π, 3π).
    """
    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    x_train = np.linspace(-2 * np.pi, 2 * np.pi, num_train_datapoints, endpoint=False)
    x_test = np.linspace(2 * np.pi, 3 * np.pi, num_test_datapoints, endpoint=False)
    y_train, y_test = np.sin(x_train), np.sin(x_test)

    return x_train, y_train, x_test, y_test


def generate_square_interpolation_dataset(gap_size, num_train_datapoints=10, num_test_datapoints=None):
    """Generate data points for a square 'bump' (_---_)."""
    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    head = np.linspace(0, 3.33, num_train_datapoints // 3, endpoint=False)
    train_bump = np.linspace(3.33, 6.66, num_train_datapoints // 3, endpoint=False)
    train_tail = np.linspace(9.99+gap_size, 13.32+gap_size, num_train_datapoints // 3, endpoint=False)

    test_bump = np.linspace(6.66, 6.66+(3.33+gap_size)/2, num_test_datapoints // 2, endpoint=False)
    test_tail = np.linspace(6.66+(3.33+gap_size)/2, 9.99+gap_size, num_test_datapoints // 2, endpoint=False)

    x_train = np.concatenate((head, train_bump, train_tail))
    x_test = np.concatenate((test_bump, test_tail))
    y_train = np.concatenate((np.zeros(head.shape[0]),
                              np.ones(train_bump.shape[0]),
                              np.zeros(train_tail.shape[0])))
    y_test = np.concatenate((np.ones(test_bump.shape[0]),
                             np.zeros(test_tail.shape[0])))
    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_interpolation_dataset(gap_size, num_train_datapoints=10, num_test_datapoints=None):
    """Generate data points for a polynomial spline."""

    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    squared_train = np.linspace(-2, gap_size, num_train_datapoints // 2, endpoint=False)
    cubed_train = np.linspace(1, 3, num_train_datapoints // 2, endpoint=False)
    x_train = np.concatenate((squared_train, cubed_train))
    y_train = np.concatenate((squared_train**2, cubed_train**3-cubed_train))

    x_test = np.linspace(gap_size, 1, num_test_datapoints, endpoint=False)
    y_test = x_test**3 - x_test

    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_extrapolation_dataset(num_train_datapoints, num_test_datapoints=None):
    """Generate data points for extrapolating a polynomial spline function."""

    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    squared_train = np.linspace(-2, 0, num_train_datapoints // 2, endpoint=False)
    cubed_train = np.linspace(0, 2, num_train_datapoints // 2, endpoint=False)
    x_train = np.concatenate((squared_train, cubed_train))
    y_train = np.concatenate((squared_train ** 2, cubed_train ** 3 - cubed_train))

    x_test = np.linspace(2, 4, num_test_datapoints, endpoint=False)
    y_test = x_test ** 3 - x_test

    return x_train, y_train, x_test, y_test


def generate_chebyshev_polynomial_interpolation_dataset(gap_size, num_train_datapoints=10, num_test_datapoints=None):
    """Generate data points for the 4th Chebyshev polynomial (16x^4 - 12x^2 + 1)."""

    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    x_train = np.concatenate((np.linspace(-1, gap_size, num_train_datapoints // 2, endpoint=False), np.linspace(0.6, 1, num_train_datapoints // 2, endpoint=False)))
    x_test = np.linspace(gap_size, 0.6, num_test_datapoints, endpoint=False)

    y_train = chebyshev_polynomial(x_train, 4)
    y_test = chebyshev_polynomial(x_test, 4)

    return x_train, y_train, x_test, y_test

def generate_parabola(gap_size, num_train_datapoints=10, num_test_datapoints=None):
    if num_test_datapoints is None:
        num_test_datapoints = num_train_datapoints // 2

    x_train = np.concatenate((np.linspace(-2, -1, num_train_datapoints, endpoint=False),
                              np.linspace(1, 2, num_train_datapoints, endpoint=False)))
    x_test = np.linspace(-1, 1, num_test_datapoints, endpoint=False)

    y_train = x_train**2
    y_test = x_test ** 2

    return x_train, y_train, x_test, y_test

# TODO: Add constant fn dataset, linear fn dataset, king's lol


def glue_dataset_portions(x_train, y_train, x_test, y_test):
    """This patches together the portions of the dataset such that we can
    apply the solution to the variational problem to all the data points."""
    tr = list(zip(x_train, y_train))
    te = list(zip(x_test, y_test))
    all = np.array(sorted(tr + te))
    x_all = all[:, 0]
    y_all = all[:, 1]
    return x_all, y_all
