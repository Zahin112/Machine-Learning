import pandas as pd
import numpy as np


def shuffler(arr, a, n):
    # We will Start from the last element
    # and swap one by one.
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i )
        print(i,j)
        arr[i], arr[j] = arr[j], arr[i]
        a[i], a[j] = a[j], a[i]

    return arr, a


def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv(r'data_banknote_authentication.csv')
    print(data.shape)
    # np.random.shuffle(data)
    # print(data)
    # X = pd.dataframe(data, columns=['variance'])
    X = data.drop('isoriginal', axis='columns').values
    # print(X)
    # y = pd.dataframe(data, columns=['isoriginal'])
    y = data.isoriginal.values
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    n = int(test_size * X.shape[0] / 100)  # shape[0]is same as len
    print(n)
    if shuffle is True:
        X, y = shuffler(X, y, len(X))
    # shuffler(X,y) is enough no assignment needed, check
    X_train = X[:n]  # 80% of X
    y_train = y[:n]  # 80% of y
    X_test = X[n:]  # 20% of X
    y_test = y[n:]  # 20% of y

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y, n):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = [], []

    # get sample size
    size = X.shape[0]
    # get list of row indexes, which indexes will be chosen
    idx = [i for i in range(size)]
    # loop through the required number of bootstraps
    for i in range(n):
        # obtain boostrap samples with replacement
        sample = np.random.choice(idx, replace=True, size=size)
        X_sample.append(X[sample, :])
        y_sample.append(y[sample])

    # assert X_sample.shape == X.shape
    # assert y_sample.shape == y.shape

    return X_sample, y_sample


# x, y = load_dataset()
# print(x.shape)
# X_train, y_train, X_test, y_test = split_dataset(x, y, 80, False)
# print(X_train.shape)
# print(X_train[:9])
# print(y_train.shape)
# X_train, y_train, X_test, y_test = split_dataset(x, y, 80, True)
# print(X_train.shape)
# print(X_train[:9])