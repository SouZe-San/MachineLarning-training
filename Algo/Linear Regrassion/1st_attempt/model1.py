
# @  Liner Regression Model create by Using Py


import numpy as np

# This is our Regression model that take no of iteration and growth_rate


class Liner_Regression:

    def __init__(self, growth_rate, iteration_no) -> None:
        self.growth_rate = growth_rate
        self.iteration_no = iteration_no

    def fit(self, X, Y):
        # number of column and number of row
        # n is number of column except Y(target) and m is number of row
        # number of training examples and number of features
        self.m, self.n = X.shape
        # initial weight and bias
        # create a array of zero with size of n*n matrix
        self.w = np.zeros(self.n)
        print(self.w)
        self.b = 0  # initial bias
        self.X = X
        self.Y = Y

        # Implementing Gradient Descent
        for i in range(self.iteration_no):
            self.update_weight()

    def update_weight(self):

        # predict the value of Y
        predict_y = self.predict(self.X)

        # Loss function for weight and bias
        dw = -(2 / self.m) * self.X.T.dot(self.Y - predict_y)
        db = -(2 / self.m) * np.sum(self.Y - predict_y)

        # update weight and bias
        self.w = self.w - self.growth_rate * dw
        self.b = self.b - self.growth_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b
