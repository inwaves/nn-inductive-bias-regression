import numpy as np
from sklearn.linear_model import LinearRegression

from utils.maths import linear


# noinspection SpellCheckingInspection
class DataAdjuster:
    def __init__(self, x, y, x_min=None, x_max=None):
        self.residual = None
        self.linear_regressor = None
        self.x = x
        self.y = y
        self.x_min = np.min(self.x) if x_min is None else x_min
        self.x_max = np.max(self.x) if x_max is None else x_max
        self.adjusted = False
        self.normalised = False

    def adjust(self):
        if self.adjusted == False and len(self.y) > 0:
            self.linear_regressor = LinearRegression().fit(self.x.reshape(-1, 1), self.y.reshape(-1, 1))
            self.residual = self.y - self.linear_regressor.predict(self.x.reshape(-1, 1)).reshape(-1)
            self.y = self.residual
            self.adjusted = True

    def unadjust(self):
        if self.adjusted == True and len(self.y) > 0:
            self.y = self.y + linear(self.x, self.linear_regressor.intercept_, self.linear_regressor.coef_[0])
            self.adjusted = False

    def normalise(self):
        if self.normalised == False and len(self.x) > 0:
            self.x = ((2 * (self.x - self.x_min)) / (self.x_max - self.x_min)) - 1
            self.normalised = True

    def unnormalise(self):
        if self.normalised == True and len(self.x) > 0:
            self.x = (((self.x_max - self.x_min) * (self.x + 1)) / 2) + self.x_min
            self.normalised = False
