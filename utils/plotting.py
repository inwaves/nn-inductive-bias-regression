import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_data(x_train, y_train, x_test, y_test, preds, x_all):
    """
    Plot the data and predictions.
    """
    # sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')
    ax.plot(x_test, y_test, 'bo', label='unseen data')
    ax.plot(x_all, preds, label='nn')
    ax.plot(x_all, np.sin(x_all), label='sin(x)')
    ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='blue')
    plt.legend(loc=(1.04, 0.5))
    plt.show()
