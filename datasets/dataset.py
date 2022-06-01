from utils.maths import *


def generate_random_dataset(task_type="baseline", num_datapoints=10):
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([0.00626232, 0.03851067, 0.15995237, 0.24668581, 0.36837388,
                            0.44915676, 0.47291526, 0.56167672, 0.95758722, 0.97322756])
        y_train = np.array([0.03480159, 0.11098908, 0.18008615, 0.18644625, 0.26021901,
                            0.36930498, 0.3823352, 0.43557663, 0.47847854, 0.60477822])

        x_test = np.array([])
        y_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([0.00626232, 0.03851067, 0.15995237, 0.47291526, 0.56167672, 0.95758722, 0.97322756])
        y_train = np.array([0.03480159, 0.11098908, 0.18008615, 0.3823352, 0.43557663, 0.47847854, 0.60477822])

        x_test = np.array([0.24668581, 0.36837388,
                           0.44915676])
        y_test = np.array([0.18644625, 0.26021901,
                           0.36930498])
    elif task_type == "extrapolation":
        x_train = np.array([0.00626232, 0.03851067, 0.15995237, 0.24668581, 0.36837388,
                            0.44915676, 0.47291526])
        y_train = np.array([0.03480159, 0.11098908, 0.18008615, 0.18644625, 0.26021901,
                            0.36930498, 0.3823352])

        x_test = np.array([0.56167672, 0.95758722, 0.97322756])
        y_test = np.array([0.43557663, 0.47847854, 0.60477822])

    return x_train, y_train, x_test, y_test


def generate_constant_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([0, 1, 2, 3, 7, 8, 9])
        x_test = np.array([4, 5, 6])
        additional_x_train = np.concatenate(
                (np.linspace(np.min(x_train), np.min(x_test),
                             int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                             endpoint=False).astype(np.float16),
                 np.linspace(np.max(x_test), np.max(x_train),
                             int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                             endpoint=False).astype(np.float16))
        )
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                        endpoint=False).astype(np.float16)
        x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([0, 1, 2, 3, 4, 5, 6])
        x_test = np.array([7, 8, 9])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                         int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
            np.float16)
        x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
        additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                        endpoint=False).astype(np.float16)
        x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train, y_test = constant(x_train), constant(x_test)
    return x_train, y_train, x_test, y_test


def generate_linear_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([0, 1, 2, 3, 7, 8, 9])
        x_test = np.array([4, 5, 6])
        additional_x_train = np.concatenate(
                (np.linspace(np.min(x_train), np.min(x_test),
                             int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                             endpoint=False).astype(np.float16),
                 np.linspace(np.max(x_test), np.max(x_train),
                             int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                             endpoint=False).astype(np.float16))
        )
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                        endpoint=False).astype(np.float16)
        x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([0, 1, 2, 3, 4, 5, 6])
        x_test = np.array([7, 8, 9])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                         int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
            np.float16)
        x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
        additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                        endpoint=False).astype(np.float16)
        x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train, y_test = linear(x_train), linear(x_test)

    return x_train, y_train, x_test, y_test


def generate_sine_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
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
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
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
        if num_datapoints > 11:
            additional_x_train = np.concatenate(
                    (np.linspace(np.min(x_train), np.min(x_test),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16),
                     np.linspace(np.max(x_test), np.max(x_train),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16))
            )
            x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
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
        if num_datapoints > 11:
            additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                             int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False)\
                .astype(np.float16)
            x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train, y_test = sin(x_train), sin(x_test)

    return x_train, y_train, x_test, y_test


def generate_square_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    """Generate data points for a square 'bump' (_---_)."""
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([i for i in range(0, 10)])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([0, 1, 2, 3, 7, 8, 9])
        x_test = np.array([4, 5, 6])
        if num_datapoints > 11:
            additional_x_train = np.concatenate(
                    (np.linspace(np.min(x_train), np.min(x_test),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16),
                     np.linspace(np.max(x_test), np.max(x_train),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16))
            )
            x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([0, 1, 2, 3, 4, 5, 6])
        x_test = np.array([7, 8, 9])
        if num_datapoints > 11:
            additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                             int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
                np.float16)
            x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train = np.array([square(el) for el in x_train])
    y_test = np.array([square(el) for el in x_test])

    return x_train, y_train, x_test, y_test


def generate_polynomial_spline_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    """Generate data points for a polynomial spline."""

    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([-2, -1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 2])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([-2, -1.75, -1.5, -1.25, 1.5, 1.75, 2, ])
        x_test = np.array([-1, 0, 1, 1.25])
        if num_datapoints > 11:
            additional_x_train = np.concatenate(
                    (np.linspace(np.min(x_train), np.min(x_test),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16),
                     np.linspace(np.max(x_test), np.max(x_train),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16))
            )
            x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([-2, -1.75, -1.5, -1.25, -1, 1, 1.25])
        x_test = np.array([1.5, 1.75, 2])
        if num_datapoints > 11:
            additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                             int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
                np.float16)
            x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train = np.array([polynomial_spline(el) for el in x_train])
    y_test = np.array([polynomial_spline(el) for el in x_test])

    return x_train, y_train, x_test, y_test


def generate_chebyshev_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    """Generate data points for the 4th Chebyshev polynomial (16x^4 - 12x^2 + 1)."""
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
    elif task_type == "interpolation":
        x_train = np.array([-1, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1])
        x_test = np.array([-0.2, 0, 0.2])
        if num_datapoints > 11:
            additional_x_train = np.concatenate(
                    (np.linspace(np.min(x_train), np.min(x_test),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16),
                     np.linspace(np.max(x_test), np.max(x_train),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16))
            )
            x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4])
        x_test = np.array([0.6, 0.8, 1])
        if num_datapoints > 11:
            additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                             int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
                np.float16)
            x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))

    y_train = np.array([chebyshev_polynomial(el, 4) for el in x_train]).reshape(1, -1).squeeze()
    y_test = np.array([chebyshev_polynomial(el, 4) for el in x_test]).reshape(1, -1).squeeze()

    return x_train, y_train, x_test, y_test


def generate_parabola_dataset(task_type="baseline", num_datapoints=10, x_test_size=0.3):
    """Generate data points for the parabola."""
    x_train, y_train, x_test, y_test = None, None, None, None
    if task_type == "baseline":
        x_train = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        additional_x_train = np.linspace(np.min(x_train), np.max(x_train), num_datapoints - len(x_train),
                                         endpoint=False).astype(np.float16)
        x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
        x_test = np.array([])
        print(type(x_train), type(x_test))
    elif task_type == "interpolation":
        x_train = np.array([-5, -4, -3, -2, 2, 3, 4, 5])
        x_test = np.array([-1, 0, 1])
        if num_datapoints > 11:
            additional_x_train = np.concatenate(
                    (np.linspace(np.min(x_train), np.min(x_test),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16),
                     np.linspace(np.max(x_test), np.max(x_train),
                                 int(((1 - x_test_size) * num_datapoints - len(x_train)) // 2),
                                 endpoint=False).astype(np.float16))
            )
            x_train = np.array(sorted(np.unique(np.append(x_train, additional_x_train))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique(np.append(x_test, additional_x_test))))
    elif task_type == "extrapolation":
        x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
        x_test = np.array([3, 4, 5])
        if num_datapoints > 11:
            additional_x_train = np.linspace(np.min(x_train), np.max(x_train),
                                             int((1 - x_test_size) * num_datapoints - len(x_train)), endpoint=False).astype(
                np.float16)
            x_train = np.array(sorted(np.unique((np.append(x_train, additional_x_train)))))
            additional_x_test = np.linspace(np.min(x_test), np.max(x_test), int(x_test_size * num_datapoints - len(x_test)),
                                            endpoint=False).astype(np.float16)
            x_test = np.array(sorted(np.unique((np.append(x_test, additional_x_test)))))
    y_train, y_test = parabola(x_train), parabola(x_test)

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
