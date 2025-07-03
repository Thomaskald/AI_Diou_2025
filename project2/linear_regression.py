import numpy as np

class LinearRegression:

    def __init__(self):

        self.w = None
        self.b = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be of type np.ndarray")
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must have 2 dimensions and y must have 1 dimension")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        N, p = X.shape

        X_aug = np.column_stack((X, np.ones(N)))

        XT_X = np.dot(X_aug.T, X_aug)
        XT_X_inv = np.linalg.inv(XT_X)
        XT_y = np.dot(X_aug.T, y)
        theta = np.dot(XT_X_inv, XT_y)

        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("The model has not been trained")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be of type np.ndarray")
        if X.ndim != 2:
            raise ValueError("X must have 2 dimensions")

        return np.dot(X, self.w) + self.b

    def evaluate(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be of type np.ndarray")
        if y.ndim != 1:
            raise ValueError("y must have 1 dimension")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")


        y_pred = self.predict(X)
        mse = (1/X.shape[0]) * np.dot((y_pred - y).T, (y_pred - y))
        return y_pred, mse