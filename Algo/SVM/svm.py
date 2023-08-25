import numpy as np


class support_vector_machine:

    def __init__(self, learning_rate, iterationNo, lambda_var) -> None:
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.iterationNo = iterationNo
        self.lambda_var = lambda_var

    def fit(self, x, y):

        self.no_sample, self.no_features = x.shape

        self.X = x
        self.Y = y
        self.w = np.zeros(self.no_features)
        self.b = 0

        for _ in range(self.iterationNo):
            self.params_update()

    def params_update(self):

        # Label encoding (cause it only recognize as 1 & -1)
        y_label = np.where(self.Y <= 0, -1, 1)  # label 0 as -1

        # gradient decent optimize
        for index, x_i in enumerate(self.X):
            # Check for gradient condition
            condition = y_label[index]*(np.dot(x_i, self.w) - self.b) >= 1

            if (condition):
                dw = 2*self.lambda_var*self.w  # dj/dw = 2λw
                db = 0
            else:
                #  dj/dw = 2λw - Yi*xi
                dw = 2 * self.lambda_var*self.w - np.dot(y_label[index], x_i)
                #  dj/db = Yi
                db = y_label[index]

            # update params
            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db

    def predict(self, X):
        y = np.dot(X, self.w) - self.b
        y_sign = np.sign(y)
        # if it -1 then return 0 otherwise 1
        y_label = np.where(y_sign <= -1, 0, 1)
        return y_label
