import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20,random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.2,random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color = 'b', marker = 'o', s=30)
# plt.show()

from linear_regression import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train,learning_rate=0.01)
predicted = regressor.predict(X_test)

mse_value = mse(y_test,predicted)
print(f"MSE on y_test = {mse_value}")

accu = r2_score(y_test,predicted)
print("Linear Regression accuracy: %f" % accu)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color='red', s=10)
m2 = plt.scatter(X_test, y_test, color='blue', s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()