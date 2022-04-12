import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

with open("logs/baseline_errors.txt", "r") as f:
    lines = f.readlines()

lines = np.array([line.replace("\n", "").split(",") for line in lines], dtype=np.float32)
average_errors = [np.mean(lines[:5, 1]), np.mean(lines[5:10, 1]), np.mean(lines[10:, 1])]


x = np.array([10, 100, 1000])

# How are these picked? Seems completely arbitrary!
factor = 3.1
exp = -0.38

plt.plot(x, average_errors)
plt.plot(x, factor * x ** exp)

plt.show()