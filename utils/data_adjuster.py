import numpy as np
from sklearn.linear_model import LinearRegression

from utils.maths import linear


class DataAdjuster:
    def __init__(self, x, y, x_min=None, x_max=None):
        self.residual = None
        self.linear_regressor = None
        self.x = x
        self.y = y
        self.x_min = np.min(self.x) if x_min is None else x_min
        self.x_max = np.max(self.x) if x_max is None else x_max

    def adjust(self):
        if len(self.y) > 0:
            self.linear_regressor = LinearRegression().fit(self.x.reshape(-1, 1), self.y.reshape(-1, 1))
            self.residual = self.y - self.linear_regressor.predict(self.x.reshape(-1, 1)).reshape(-1)
            self.y = self.residual

    def unadjust(self):
        if len(self.y) > 0:
            self.y = self.y + linear(self.x, self.linear_regressor.intercept_, self.linear_regressor.coef_[0])

    def normalise(self):
        if len(self.x) > 0:
            self.x = ((2 * (self.x - self.x_min)) / (self.x_max - self.x_min)) - 1

    def unnormalise(self):
        if len(self.x) > 0:
            self.x = (((self.x_max - self.x_min) * (self.x + 1)) / 2) + self.x_min
