import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

num_runs = 5

def linlinplot(x, average_errors):
    # How are these picked? Seems completely arbitrary!
    factor = 4
    exp = -0.45

    plt.plot(x, average_errors, label="NN error")
    plt.plot(x, factor * x ** exp, label=f"{factor} * x^{exp}")
    plt.legend()

    plt.show()


def loglogplot(x, average_errors):
    """This follows the method in the authors' code."""
    X = np.log(x).reshape(-1, 1)
    ys = np.log(average_errors)
    reg = LinearRegression().fit(X, ys)
    print(f"Regression score: {reg.score(X, ys)}")

    Xline = np.linspace(10, 5120, 1000).reshape(-1, 1)
    plt.loglog(x, average_errors, "-o")
    plt.loglog(Xline, np.exp(reg.predict(np.log(Xline))))
    plt.loglog(Xline, 1/np.sqrt(Xline))
    plt.legend([f"Error",
                f"$y={np.exp(reg.intercept_):.4f}x^{{{reg.coef_[0]:.4f}}}$",
                r"$y=\frac{1}{\sqrt{n}}$"], fontsize='large')

    plt.show()


if __name__ == '__main__':
    with open("logs/baseline_errors.txt", "r") as f:
        lines = f.readlines()

    lines = np.array([line.replace("\n", "").split(",") for line in lines], dtype=np.float32)
    average_errors = [np.mean(lines[:num_runs, 1]), np.mean(lines[num_runs:2*num_runs, 1]), np.mean(lines[2*num_runs:, 1])]
    print(f"Average errors: {average_errors}")
    x = np.array([10, 100, 1000])

    loglogplot(x, average_errors)
