import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_data(x, preds, y):
    """
    Plot the data and predictions.
    """
    plt.plot(x, y, 'ro', label='data')
    plt.plot(x, preds, label='predictions')
    plt.legend()
    plt.show()
