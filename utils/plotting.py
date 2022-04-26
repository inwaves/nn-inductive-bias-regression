import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb


def plot_data_vs_predictions(x_train, y_train, x_test, y_test, y_pred, x_all, grid, g_star_preds, fn):
    """
    Plot the data and predictions.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')

    if len(x_test) > 0:
        ax.plot(x_test, y_test, 'bo', label='unseen data')
        ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='blue')

    ax.plot(x_all, y_pred, label='nn')
    ax.plot(grid, g_star_preds, label='cubic spline')

    # Plot ground truth function.
    fn_y = [fn(el) for el in grid]
    ax.plot(grid, fn_y, label="ground truth")
    plt.legend()
    wandb.log({"plot": plt})


def plot_raw_data(x_train, y_train, x_test, y_test):
    """
    Plot the raw data.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')

    if len(x_test) > 0:
        ax.plot(x_test, y_test, 'bo', label='unseen data')
        ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='blue')
    plt.legend()
    plt.show()
