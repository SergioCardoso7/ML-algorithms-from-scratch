import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


from decision_tree import DecisionTree
from utils import accuracy


data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)


classifier = DecisionTree(max_depth=10)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(f"Decision Tree classifier accuracy: {accuracy(y_test,y_pred)}")