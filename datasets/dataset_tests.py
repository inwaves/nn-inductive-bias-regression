from datasets.dataset import *
from utils.plotting import plot_raw_data


def generate_random_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_random_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_random_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_random_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_random_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_random_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_constant_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_constant_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_constant_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_constant_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_constant_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_constant_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_linear_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_linear_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_linear_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_linear_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_linear_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_linear_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_sine_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_sine_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_sine_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_sine_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_sine_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_sine_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_square_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_square_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_square_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_square_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_square_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_square_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_polynomial_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_polynomial_spline_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_polynomial_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_polynomial_spline_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_polynomial_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_polynomial_spline_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_chebyshev_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_chebyshev_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_chebyshev_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_chebyshev_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_chebyshev_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_chebyshev_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_parabola_baseline_unit_test():
    x_tr, y_tr, x_te, y_te = generate_parabola_baseline()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_parabola_interpolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_parabola_interpolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


def generate_parabola_extrapolation_unit_test():
    x_tr, y_tr, x_te, y_te = generate_parabola_extrapolation()
    plot_raw_data(x_tr, y_tr, x_te, y_te)


if __name__ == "__main__":
    generate_linear_interpolation_unit_test()
