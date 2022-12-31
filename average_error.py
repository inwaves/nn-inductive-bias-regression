import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

num_runs = 3
# categories = ["chebyshev_extrapolation",
#               "constant_extrapolation",
#               "linear_extrapolation",
#               "parabola_extrapolation",
#               "piecewise-polynomial_extrapolation",
#               "sine_extrapolation",
#               "square_extrapolation"]
categories = ["sine_moredata"]


def loglogplot(x, average_errors, standard_deviations, category, type):
    """This follows the method in the authors' code."""
    X = np.log(x).reshape(-1, 1)
    ys = np.log(average_errors)
    reg = LinearRegression().fit(X, ys)
    print(f"Regression score: {reg.score(X, ys)}")

    Xline = np.linspace(np.min(x), (np.max(x)), 1000).reshape(-1, 1)
    plt.figure()
    plt.xlabel("Number of hidden units")
    plt.ylabel(f"{type} error")
    plt.title(f"{type} error for {category} as network scales")
    plt.loglog(x, average_errors, "-o")
    plt.errorbar(x, average_errors, yerr=standard_deviations, fmt="none", capsize=5)
    plt.loglog(Xline, np.exp(reg.predict(np.log(Xline))))
    plt.loglog(Xline, 1/np.sqrt(Xline))
    plt.legend([f"Error",
                f"$y={np.exp(reg.intercept_):.4f}x^{{{reg.coef_[0]:.4f}}}$",
                r"$y=\frac{1}{\sqrt{n}}$"], fontsize='large')

    if not os.path.exists("plots/error/"):
        os.makedirs("plots/error/")
    plt.savefig(f"plots/error/{category}-{type}.png")


if __name__ == '__main__':
    # x = np.array([10, 100, 150, 500])
    x = np.array([10, 100, 500, 1000, 10000])

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
        print(f"{category} average var_errs: {avg_var_errs}, val_errs: {avg_val_errs}")
        loglogplot(x, avg_var_errs, var_stdevs, category, "variational")
        loglogplot(x, avg_val_errs, val_stdevs, category, "validation")
        lines = lines[len(x)*num_runs:]