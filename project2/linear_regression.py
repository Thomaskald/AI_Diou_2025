import numpy as np

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

        ones = np.ones((self.N,1))
        X_new = np.hstack((X, ones))

        XT_X = X_new.T @ X_new
        XT_y = X_new.T @ y
        theta = np.linalg.inv(XT_X) @ XT_y

        self.w = theta[:-1].flatten()
        self.b = theta[-1][0]

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("The model has not been trained")

        X = np.array(X)

        y_pred = X @ self.w + self.b
        return y_pred

    def evaluate(self, X, y):
        if self.w is None or self.b is None:
            raise ValueError("The model has not been trained")

        y_pred = self.predict(X)

        mse = (y_pred - y).T @ (y_pred - y) / len(y)

        mse = mse.item()

        return y_pred.flatten(), mse