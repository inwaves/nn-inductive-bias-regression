import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import wandb

from utils.parsers import parse_bool


def plot_data_vs_predictions(x_train, y_train, x_test, y_test, y_all_pred, grid, g_star_preds, fn_y, args):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', name='training_data'))

    if len(x_test) > 0:
        fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', name='test_data'))
        fig.add_vrect(x0=np.min(x_test), x1=np.max(x_test), line_width=0, fillcolor="red", opacity=0.2)
    fig.add_trace(go.Scatter(x=grid, y=y_all_pred.reshape(grid.shape), mode='lines', name='nn'))
    fig.add_trace(go.Scatter(x=grid, y=g_star_preds.reshape(grid.shape), mode='lines', name='cubic spline'))
    fig.add_trace(go.Scatter(x=grid, y=fn_y, mode='lines', name='ground truth'))
    wandb.log({"plot": fig})

    if not os.path.exists("plots/fit/"):
        os.makedirs("plots/fit/")

    normalised = "normalised" if parse_bool(args.normalise) else "unnormalised"
    adjusted = "adjusted" if parse_bool(args.adjust_data_linearly) else "unadjusted"
    fig.write_image(f"plots/fit/{args.dataset}_{args.generalisation_task}_n={args.hidden_units}_"
                    f"{normalised}_{adjusted}.png")


def plot_raw_data(x_train, y_train, x_test, y_test):
    """
    Plot the raw data.
    """
    sns.set()
    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'bo', label='data')

    if len(x_test) > 0:
        ax.plot(x_test, y_test, 'ro', label='unseen data')
        ax.axvspan(np.min(x_test), np.max(x_test), alpha=0.1, color='red')
    plt.legend()
    plt.show()
