import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

num_runs = 3
categories = ["sine", "square", "parabola", "polynomial_spline", "chebyshev_polynomial", "constant", "linear"]


def loglogplot(x, average_errors, category):
    """This follows the method in the authors' code."""
    X = np.log(x).reshape(-1, 1)
    ys = np.log(average_errors)
    reg = LinearRegression().fit(X, ys)
    print(f"Regression score: {reg.score(X, ys)}")

    Xline = np.linspace(10, 5120, 1000).reshape(-1, 1)
    plt.figure()
    plt.title(f"Error for {category} as network scales")
    plt.loglog(x, average_errors, "-o")
    plt.loglog(Xline, np.exp(reg.predict(np.log(Xline))))
    plt.loglog(Xline, 1/np.sqrt(Xline))
    plt.legend([f"Error",
                f"$y={np.exp(reg.intercept_):.4f}x^{{{reg.coef_[0]:.4f}}}$",
                r"$y=\frac{1}{\sqrt{n}}$"], fontsize='large')

    if not os.path.exists("plots/error/"):
        os.makedirs("plots/error/")
    plt.savefig(f"plots/error/{category}.png")


if __name__ == '__main__':
    x = np.array([10, 100, 500, 1000, 5000])

    with open("logs/baseline_errors.txt", "r") as f:
        lines = f.readlines()

    lines = np.array([line.replace("\n", "").split(",")[1:] for line in lines], dtype=np.float32)
    for category in categories:
        average_errors = []
        for i, _ in enumerate(x):
            average_errors.append(np.mean(lines[i*num_runs: (i+1)*num_runs, 1]))
        print(f"{category} average errors: {average_errors}")
        loglogplot(x, average_errors, category)
        lines = lines[len(x)*num_runs:]

