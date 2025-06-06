import numpy
import numpy as np

class LinearRegression:
    def __init__(self):

        self.w = None
        self.b = None
        self.N = None
        self.p = None

    def fit(self, X, y):
        X = numpy.array(X)
        y = numpy.array(y)

        if len(X) != len(y):
            raise ValueError("X and y must be compatible")