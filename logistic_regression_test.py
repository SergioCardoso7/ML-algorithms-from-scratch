import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import logisticRegression
import utils as ut

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
    

bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.2,random_state=1234)

my_threshold = 0.5

X_train = ut.z_score_norm(X_train)
X_test = ut.z_score_norm(X_test)

regressor = logisticRegression()
regressor.fit(X_train, y_train,learning_rate=1.8)
predictions = regressor.predict(X_test,my_threshold)

print(f"Logistic Regression accuracy: {accuracy(y_test,predictions)}")