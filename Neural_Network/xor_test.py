"""
Test to see if the neural network can emulate the xor function
xor is not linearly separable
the neural network will need to find a non-linear function to solve this problem
"""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import mse_loss, mse_loss_prime
import numpy as np
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

for i in range(epochs):
    error = 0
    for x,  y in zip(X,Y):
        
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        error += mse_loss(y, output)
        
        # backward 
        grad = mse_loss_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(X)
    if i % ceil(epochs / 10) == 0 or i == epochs - 1:
        print(f"{i + 1}/{epochs}, error = {error:f}")