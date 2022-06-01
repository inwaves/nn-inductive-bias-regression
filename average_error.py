import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

num_runs = 2
# categories = ["chebyshev", "constant", "linear", "parabola", "piecewise-polynomial", "sine", "square"]
categories = ["piecewise-polynomial"]


def loglogplot(x, average_errors, standard_deviations, category, plot_type):
    """This follows the method in the authors' code."""
    X = np.log(x).reshape(-1, 1)
    ys = np.log(average_errors)
    reg = LinearRegression().fit(X, ys)
    print(f"Regression score: {reg.score(X, ys)}")
    fig, ax = plt.subplots()

    Xline = np.linspace(np.min(x), (np.max(x)), 1000).reshape(-1, 1)
    plt.figure()
    plt.title(f"Error for {category} as network scales")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.loglog(x, np.sqrt(average_errors), "-o")
    print(f"Error bars: {np.sqrt(standard_deviations) / np.sqrt(average_errors)/2}")
    ax.errorbar(x, np.sqrt(average_errors), yerr=np.sqrt(standard_deviations) / np.sqrt(average_errors)/2, fmt="none", capsize=5)
    ax.loglog(Xline, np.exp(reg.predict(np.log(Xline))))
    ax.loglog(Xline, 1/np.sqrt(Xline))
    ax.legend([f"Error",
                f"$y={np.exp(reg.intercept_):.4f}x^{{{reg.coef_[0]:.4f}}}$",
                r"$y=\frac{1}{\sqrt{n}}$"], fontsize='large')

    if not os.path.exists("plots/error/"):
        os.makedirs("plots/error/")
    fig.savefig(f"plots/error/{category}-{plot_type}.png")


if __name__ == '__main__':
    x = np.array([10, 100, 150, 500])
    # x = np.array([10, 100, 500, 1000, 10000])

    with open("logs/baseline_errors.txt", "r") as f:
        lines = f.readlines()

    lines = np.array([line.replace("\n", "").split(",")[1:] for line in lines], dtype=np.float32)
    for category in categories:
        avg_var_errs, avg_val_errs, var_stdevs, val_stdevs = [], [], [], []
        for i, _ in enumerate(x):
            avg_var_errs.append(np.mean(lines[i * num_runs: (i + 1) * num_runs, 1]))
            var_stdevs.append(np.std(lines[i * num_runs: (i + 1) * num_runs, 1]))
            avg_val_errs.append(np.mean(lines[i * num_runs: (i + 1) * num_runs, 2]))
            val_stdevs.append(np.std(lines[i * num_runs: (i + 1) * num_runs, 2]))
        print(f"{category} average var_errs: {avg_var_errs}\n val_errs: {avg_val_errs}")
        print(f"{category} var_stdevs: {var_stdevs}\n val_stdevs: {val_stdevs}")
        loglogplot(x, avg_var_errs, var_stdevs, category, "variational")
        loglogplot(x, avg_val_errs, val_stdevs, category, "validation")
        lines = lines[len(x)*num_runs:]
