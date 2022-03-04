import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_sin_data_vs_predictions(x_train, y_train, x_test, y_test, preds, x_all, y_all):
    """
    Plot the data and predictions.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')
    ax.plot(x_test, y_test, 'bo', label='unseen data')
    ax.plot(x_all, preds, label='nn')
    ax.plot(x_all, y_all, label='ground truth')
    ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='blue')
    plt.legend()
    plt.show()


def plot_raw_data(x_train, y_train, x_test, y_test):
    """
    Plot the raw data.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')
    ax.plot(x_test, y_test, 'bo', label='unseen data')
    ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='blue')
    plt.legend()
    plt.show()
