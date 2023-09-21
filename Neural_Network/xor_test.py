"""
Test to see if the neural network can emulate the xor function
xor is not linearly separable
the neural network will need to find a non-linear function to solve this problem
"""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import mse_loss, mse_loss_prime, train, predict, accuracy
import numpy as np
import matplotlib.pyplot as plt
from dense_layer import Dense
from activations import Tanh
from math import ceil


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]],(4, 2, 1))
Y = np.reshape([[0],[1],[1],[0]], (4, 1, 1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1

train(network,mse_loss,mse_loss_prime,X,Y,epochs,learning_rate)

points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()