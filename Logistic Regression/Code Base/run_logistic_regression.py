"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score

if __name__ == '__main__':
    # data load
    X, y = load_dataset()

    # split train and test
    #np.random.shuffle(arr)
    X_train, y_train, X_test, y_test = split_dataset(X, y, 80, False)

    # training
    params = dict()
    classifier = LogisticRegression(rate=0.01)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)
    #
    # # performance on test set
    tp = tn = fp = fn = 0
    for actual, predicted in zip(y_test, y_pred):
        if predicted == 1:
            if predicted == actual:
                tp += 1
            else:
                fp += 1
        else:
            if predicted == actual:
                tn += 1
            else:
                fn += 1

    prec = precision_score(tp=tp, fp=fp)
    recall = recall_score(tp=tp, fn=fn)

    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall)
    print('Precision score ', prec)
    print('F1 score ', f1_score(prec=prec, recall=recall))
