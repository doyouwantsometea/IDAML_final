from sklearn.linear_model import LinearRegression



vanilla_reg = LinearRegression()

vanilla_reg.fit(X_train, y_train)
y_pred = vanilla_reg.predict(X_test)
vanilla_reg.score(X_test, y_test)