import numpy as np

# this is the function that create the S carve


def sigmoid(x):

    return 1/(1 + np.exp(-x))

# Create the class --


class LogisticRegression:
    def __init__(self, no_iteration, learning_rate) -> None:
        self.w = None
        self.b = None
        self.no_iteration = no_iteration
        self.learning_rate = learning_rate

    # Fit the data --

    def fit(self, X, Y):

        # get the number of features --
        self.no_samples, self.no_features = X.shape

        self.w = np.zeros(self.no_features)
        self.b = 0
        self.X = X
        self.Y = Y

        # iterate the learning process using  Gradient Descent for Optimization
        for _ in range(self.no_iteration):
            self.update_parm()

    def update_parm(self):

        z = np.dot(self.X, self.w) + self.b  # z = w*X +b
        # Get Final y using sigmoid
        predictions_y = sigmoid(z)

        # calculation error
        dw = (1/self.no_samples) * np.dot(self.X.T, (predictions_y - self.Y))
        db = (1/self.no_samples) * np.sum(predictions_y - self.Y)

        #  updating the weights & bias using gradient descent

        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, x):
        z = z = np.dot(x, self.w) + self.b
        predict_y = sigmoid(z)
        predict_answer = [0 if y <= 0.5 else 1 for y in predict_y]
        return predict_answer
