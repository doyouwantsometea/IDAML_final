from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def baseline_model(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print(f"Score:{reg.score(X_test, y_test)}")

def tune_model(X_train, X_test, y_train, y_test):
    regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())
    regressor.fit(X_train, y_train)

    # y_pred = regressor.predict(X_test)
    print(f"Score:{regressor.score(X_test, y_test)}")