import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
from linear_regression import LinearRegression

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# My implementation
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
_, mse = linear_reg.evaluate(X_test, y_test)
print(f"RMSE from my implementation: {np.sqrt(mse)}")

# scikit-learn implementation
sk_linear_reg = SklearnLinearRegression()
sk_linear_reg.fit(X_train, y_train)
y_pred = sk_linear_reg.predict(X_test)
sk_mse = mean_squared_error(y_test, y_pred)
print(f"RMSE from scikit-learn implementation: {np.sqrt(sk_mse)}")

# For 20 repetitions
my_rmse = []
sk_rmse = []

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # My implementation
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    _, mse = linear_reg.evaluate(X_test, y_test)
    my_rmse.append(np.sqrt(mse))

    # scikit-learn implementation
    sk_linear_reg = SklearnLinearRegression()
    sk_linear_reg.fit(X_train, y_train)
    y_pred = sk_linear_reg.predict(X_test)
    sk_mse = mean_squared_error(y_test, y_pred)
    sk_rmse.append(np.sqrt(sk_mse))

print("\n20 repetitions")
print(f"My implementation - Mean RMSE: {np.mean(my_rmse)}, Standard Deviation: {np.std(my_rmse)}")
print(f"Sickit-learn - Mean RMSE: {np.mean(sk_rmse)}, Standard Deviation: {np.std(sk_rmse)}")