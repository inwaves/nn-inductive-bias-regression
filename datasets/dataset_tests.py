from datasets.dataset import generate_square_interpolation, generate_chebyshev_interpolation
from utils.plotting import plot_raw_data

gap_size = 0


def generate_square_interpolation_dataset_unit_test():
    x_tr, y_tr, x_te, y_te = generate_square_interpolation(gap_size)
    plot_raw_data(x_tr, y_tr, x_te, y_te)

def generate_chebyshev_polynomial_interpolation_dataset_unit_test():
    x_tr, y_tr, x_te, y_te = generate_chebyshev_interpolation(gap_size, num_train_datapoints=50)
    plot_raw_data(x_tr, y_tr, x_te, y_te)

if __name__ == "__main__":
    generate_chebyshev_polynomial_interpolation_dataset_unit_test()
