import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, rate, n=2000):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.rate = rate
        self.n = n
        self.weight = self.bias = None

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        # print(X.shape)
        sample, feature = X.shape
        self.weights = np.zeros(feature)
        self.bias = 0
        # costlist = []

        for _ in range(self.n):
            pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(pred)
            # cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

            dw = (1 / sample) * np.dot(X.T, (predictions - y))
            db = (1 / sample) * np.sum(predictions - y)

            self.weights = self.weights - self.rate * dw
            self.bias = self.bias - self.rate * db
            # cost_list.append(cost)

        return self

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(pred)
        predict = [0 if y < 0.5 else 1 for y in y_pred]

        return predict
