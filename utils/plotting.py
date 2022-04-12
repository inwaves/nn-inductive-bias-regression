import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb


def plot_sin_data_vs_predictions(x_train, y_train, x_test, y_test, y_pred, x_all, grid, g_star_preds):
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
    ax.plot(grid, np.sin(2*np.pi*grid), label="sin ground truth")
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
