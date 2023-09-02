#  Lasso Regression Also known as L1 Regularization
# This Is use when the Regression become overfitted ,
import numpy as np


class lasso_Regression:

    def __init__(self, iteration, learning_rate, lambda_pram) -> None:
        self.iteration = iteration
        self.learningRate = learning_rate
        self.lambdaParam = lambda_pram
        self.w = None
        self.b = None

    def fit(self, x, y):
        # get Row and colum of input data
        self.no_sample, self.no_features = x.shape

        self.w = np.zeros(self.no_features)
        self.b = 0

        self.X = x
        self.Y = y

        # Start reduce the loss
        for _ in range(self.iteration):
            self.params_update()

    def params_update(self):

        # Get Predict y
        pred_y = self.predict(self.X)

        # get dw matrix filled with 0
        dw = np.zeros(self.no_features)

        # Σ of 1 to n
        for i in range(self.no_features):
            if self.w[i] > 0:
                dw[i] = (-(2*(self.X[:, i]).dot(self.Y - pred_y)) +
                         self.lambdaParam) / self.no_sample
            else:
                dw[i] = (-2/self.no_sample) * ((self.X[:, i]).dot
                                               (self.Y - pred_y) - self.lambdaParam)

        db = (-2/self.no_sample) * np.sum(self.Y - pred_y)

        # updates weight and bias
        self.w = self.w - self.learningRate*dw
        self.b = self.b - self.learningRate*db

    def predict(self, x):
        pred_y = np.dot(x, self.w) + self.b
        # print(pred_y)

        return pred_y
