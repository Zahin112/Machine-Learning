
from data_handler import bagging_sampler
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.models = []

    def fit(self, X_train, y_train):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X_train.shape[0] == y_train.shape[0]
        assert len(X_train.shape) == 2
        # todo: implement
        # make bootstrap samples
        x_sample, y_sample = bagging_sampler(X_train, y_train, self.n_estimator)
        size = self.n_estimator
        # iterate through each bootstrap sample & fit a model ##
        for b in range(size):
            model = self.base_estimator.fit(x_sample[b], y_sample[b])
            # append the fitted model
            self.models.append(model)

        return self


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        predictions = []
        for m in self.models:
            # make predictions on the input X
            yp = m.predict(X)
            #print(yp)
            # append predictions to storage list
            predictions.append(np.array(yp).reshape(-1, 1))
        # compute the ensemble prediction
        #print(predictions)
        y_pred = np.round(np.mean(np.concatenate(predictions, axis=1), axis=1)).astype(int)
        # y_pred = [0 if np.mean(y) < 0.5 else 1 for y in predictions]
        #print(y_pred)
        # return the prediction
        return y_pred
