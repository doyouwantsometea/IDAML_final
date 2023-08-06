import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import explained_variance_score, mean_squared_error, accuracy_score



# class GDLinearRegression:
#     def __init__(self, lr=0.01, iter=1000):
#         self.lr = lr
#         self.iter = iter
    
#     def fit(self, X, y):
#         b = 0
#         m = 5
#         n = X.shape[0]
#         for _ in range(self.iter):
#             b_gradient = -2 * np.sum(y - m*X + b) / n
#             m_gradient = -2 * np.sum(X*(y - (m*X + b))) / n
#             b = b + (self.lr * b_gradient)
#             m = m - (self.lr * m_gradient)
#         self.m, self.b = m, b
        
#     def predict(self, X):
#         return self.m*X + self.b
    
#     def score(self, X, y):
#         y_pred = self.predict(X)
#         u = ((y - y_pred)** 2).sum()
#         v = ((y - y.mean()) ** 2).sum()
#         print(1 - u/v)
#         return 1 - u/v



def baseline_model(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print(f"Score:{reg.score(X_test, y_test)}")


def classification(X_train, X_test, y_train, y_test):

    next_X_train = X_train.where(X_train['label'] > 0).dropna()    
    # print(next_X_train)
    next_y_train = y_train.loc[next_X_train.index]
    # print(next_y_train)


    y_train = X_train['label']
    X_train = X_train.drop('label', axis=1)
    # print(y_train)
    # print(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test.drop('label', axis=1))
    X_test['pred_label'] = y_pred
    # print(X_test.head(10))
    # print(X_test.where(y_pred == 1))
    # print(y_pred.tolist())
    # print(X_test['label'].tolist())
    print(f'LogReg Accuracy: {accuracy_score(y_pred.tolist(), X_test.label.tolist())}')

    # next_X = X_train.where(X_train['label'] > 0)
    # print(next_X)
    # next_X_test = X_test.where(X_test['pred_label'] > 0).dropna().drop(['pred_label'], axis=1)
    next_X_test = X_test
    # print(next_X_test)
    # next_y_test = y_test.loc[next_X_test.index]
    next_y_test = y_test
    # print(next_y_test)
    return next_X_train, next_X_test, next_y_train, next_y_test


def single_model(X_train, X_test, y_train, y_test, add_layer=False):
    # regressor = make_pipeline(LinearRegression())
    # regressor = KernelRidge(degree=4)
    # regressor = Ridge()
    # regressor = ElasticNet(alpha=0.1)
    # regressor = GDLinearRegression()
    
    if add_layer:
        pred = X_test['pred_label']
        X_test = X_test.where(X_test['pred_label'] > 0).dropna().drop(['pred_label'], axis=1)
        y_test = y_test.loc[X_test.index]

    param_grid = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
        # "loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        # "penalty": ["l2", "l1", "elasticnet"],
                  }

    # regressor = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
    # regressor = DecisionTreeRegressor(min_samples_leaf=10)
    regressor = KernelRidge(kernel="linear")
    regressor.fit(X_train, y_train)

    # search = GridSearchCV(regressor, param_grid)    
    # search.fit(X_train, y_train)
    # best_regressor = search.best_estimator_

    y_pred = regressor.predict(X_test)
    # print(y_test[:10])
    # print(y_pred[:10])
    print(f"KernelRidge Score: {regressor.score(X_test, y_test)}")
    # print(y_pred)
    # y_pred = best_regressor.predict(X_test)
    # print(f"Score:{best_regressor.score(X_test, y_test)}")

    print(explained_variance_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))

    pred.loc[pred.values > 0] += (y_pred - 1)
    return pred


def evaluate(pred, true):
    # u = ((true - pred) ** 2).sum()
    # v = ((true - true.mean()) ** 2).sum()
    # print(1 - u/v)
    print(f'Variance: {explained_variance_score(true, pred)}')
    print(f'Mean Squared Error: {mean_squared_error(true, pred)}')