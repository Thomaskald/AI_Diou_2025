import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):

        self.w = None
        self.b = None
        self.N = None
        self.p = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if len(X) != len(y):
            raise ValueError("X and y must be compatible")