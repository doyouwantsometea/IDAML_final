import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import explained_variance_score, mean_squared_error, accuracy_score


def baseline_model(X_train, X_test, y_train, y_test):
    # vanilla ridge regression
    reg = Ridge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return y_pred


def classification(X_train, X_test, y_train, y_test):
    # restructure data for ridge regression after classification
    next_X_train = X_train.where(X_train['label'] > 0).dropna()
    next_y_train = y_train.loc[next_X_train.index]

    # add label for classification
    y_train = X_train['label']
    X_train = X_train.drop('label', axis=1)

    # logistic regression classification
    clf = LogisticRegression(max_iter=1000, C=0.001)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test.drop('label', axis=1))
    X_test['pred_label'] = y_pred

    print(f'LogReg Accuracy: {accuracy_score(y_pred.tolist(), X_test.label.tolist())}')
    return next_X_train, next_y_train


def kernel_ridge(X_train, X_test, y_train, y_test, add_layer=False):

    # restructure data if logistic regression is performed
    if add_layer:
        pred = X_test['pred_label']
        X_test = X_test.where(X_test['pred_label'] > 0).dropna().drop(['pred_label'], axis=1)
        y_test = y_test.loc[X_test.index]

    regressor = KernelRidge(kernel='linear')
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    if add_layer:
        print(f'Kernel Ridge R2: {explained_variance_score(y_test, y_pred, force_finite=False)}')
        pred.loc[pred.values > 0] += (y_pred - 1)
        return pred
    
    else:
        return y_pred


def evaluate(pred, true):
    print(f'R2 Score: {explained_variance_score(true, pred, force_finite=False)}')
    print(f'MSE: {mean_squared_error(true, pred)}')