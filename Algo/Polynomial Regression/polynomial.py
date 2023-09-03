import numpy as np


class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree

    def fit_transform(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        num_samples, num_features = X.shape
        X_poly = np.ones((num_samples, 1))

        for d in range(1, self.degree + 1):
            for feature in range(num_features):
                X_poly = np.column_stack((X_poly, X[:, feature] ** d))

        return X_poly
