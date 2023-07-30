import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import explained_variance_score, mean_squared_error



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


def tune_model(X_train, X_test, y_train, y_test):
    # regressor = make_pipeline(LinearRegression())
    # regressor = KernelRidge(degree=4)
    # regressor = Ridge()
    # regressor = ElasticNet(alpha=0.1)
    # regressor = GDLinearRegression()
    
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
    print(y_test[:10])
    print(y_pred[:10])
    print(f"Score:{regressor.score(X_test, y_test)}")

    # y_pred = best_regressor.predict(X_test)
    # print(f"Score:{best_regressor.score(X_test, y_test)}")

    print(explained_variance_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))