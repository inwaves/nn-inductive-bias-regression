import numpy as np

from datasets.dataset import glue_dataset_portions
from utils.adjust_data import DataAdjuster
from utils.parsers import parse_bool


def initialise_grid(da_train, da_test, fn, adjust_data_linearly, normalise, grid_resolution):
    # Using raw data...
    adjust_data = parse_bool(adjust_data_linearly)
    normalise = parse_bool(normalise)
    if adjust_data:
        da_train.unadjust()
        da_test.unadjust()
    if normalise:
        da_train.unnormalise()
        da_test.unnormalise()
    raw_x_all, raw_y_all = glue_dataset_portions(da_train.x, da_train.y, da_test.x, da_test.y)

    # ...generate a grid with more datapoints
    grid = np.linspace(np.min(raw_x_all), np.max(raw_x_all), grid_resolution)
    fn_y = np.array([fn(el) for el in grid]).reshape(1, -1).squeeze()
    da_grid = DataAdjuster(grid, fn_y, da_train.x_min, da_train.x_max)
    if adjust_data:
        da_grid.adjust()
    if normalise:
        da_grid.normalise()
    return da_grid