import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import wandb


def plot_data_vs_predictions(x_train, y_train, x_test, y_test, x_all, y_all_pred, grid, g_star_preds, fn_y):
    """
    Plot the data and predictions.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'ro', label='data')

    if len(x_test) > 0:
        ax.plot(x_test, y_test, 'bo', label='unseen data')
        ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.3, color='blue')
        print(f"Should have just plotted a vspan. x_test: {x_test}")
    ax.plot(x_all, y_all_pred, label='nn')
    ax.plot(x_all, g_star_preds, label='cubic spline')
    ax.plot(grid, fn_y, label="ground truth")
    plt.legend()
    wandb.log({"plot": plt})
    plt.show()


def plot_data_plotly(x_train, y_train, x_test, y_test, x_all, y_all_pred, grid, g_star_preds, fn_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='training_data'))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='test_data'))
    fig.add_trace(go.Scatter(x=x_all, y=y_all_pred.reshape(x_all.shape), mode='lines', name='nn'))
    fig.add_trace(go.Scatter(x=x_all, y=g_star_preds.reshape(x_all.shape), mode='lines', name='cubic spline'))
    fig.add_trace(go.Scatter(x=grid, y=fn_y, mode='lines', name='ground truth'))
    fig.add_vrect(x0=np.min(x_test), x1=np.max(x_test), line_width=0, fillcolor="red", opacity=0.2)
    wandb.log({"plot": fig})


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
