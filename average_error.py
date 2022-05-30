import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

num_runs = 3
categories = ["chebyshev", "linear_adjusted", "parabola", "piecewise_polynomial", "sine", "square",
              "constant_unadjusted", "linear_unadjusted"]


def loglogplot(x, average_errors, standard_deviations, category):
    """This follows the method in the authors' code."""
    X = np.log(x).reshape(-1, 1)
    ys = np.log(average_errors)
    reg = LinearRegression().fit(X, ys)
    print(f"Regression score: {reg.score(X, ys)}")

    Xline = np.linspace(np.min(x), (np.max(x)), 1000).reshape(-1, 1)
    plt.figure()
    plt.title(f"Error for {category} as network scales")
    plt.loglog(x, average_errors, "-o")
    plt.errorbar(x, average_errors, yerr=standard_deviations, fmt="none", capsize=5)
    plt.loglog(Xline, np.exp(reg.predict(np.log(Xline))))
    plt.loglog(Xline, 1/np.sqrt(Xline))
    plt.legend([f"Error",
                f"$y={np.exp(reg.intercept_):.4f}x^{{{reg.coef_[0]:.4f}}}$",
                r"$y=\frac{1}{\sqrt{n}}$"], fontsize='large')

    if not os.path.exists("plots/error/"):
        os.makedirs("plots/error/")
    plt.savefig(f"plots/error/{category}.png")


if __name__ == '__main__':
    # x = np.array([10, 40, 160, 640, 10240])
    x = np.array([10, 100, 500, 1000, 10000])

    with open("logs/baseline_errors.txt", "r") as f:
        lines = f.readlines()

    lines = np.array([line.replace("\n", "").split(",")[1:] for line in lines], dtype=np.float32)
    for category in categories:
        avg_errs, st_devs = [], []
        for i, _ in enumerate(x):
            avg_errs.append(np.mean(lines[i * num_runs: (i + 1) * num_runs, 1]))
            st_devs.append(np.std(lines[i * num_runs: (i + 1) * num_runs, 1]))
        print(f"{category} average errors: {avg_errs}")
        loglogplot(x, avg_errs, st_devs, category)
        lines = lines[len(x)*num_runs:]

