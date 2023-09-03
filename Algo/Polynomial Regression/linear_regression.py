import numpy as np


class LinearRegression:
    def __init__(self, lr, iter) -> None:
        self.w = None
        self.b = None
        self.lr = lr
        self.iter = iter

    def fit(self, X, Y):
        no_sample, no_feature = X.shape
        self.w = np.zeros(no_feature)
        self.b = 0

        for _ in range(self.iter):
            y_pred = self.predict(X)

            dw = (1/no_sample) * np.dot(X.T, (y_pred - Y))
            db = (1/no_sample) * np.sum(y_pred - Y)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
