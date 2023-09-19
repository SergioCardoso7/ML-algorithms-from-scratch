from base_layer import BaseLayer
import numpy as np


class Dense(BaseLayer):
    
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(output_size, input_size)
        self.B = np.random.rand(output_size, 1)
    
    def forward(self, X):
        self.X = X
        return np.dot(self.W, self.X) + self.B
    
    def backward(self, output_gradient, learning_rate):
        #update parameters, return input gradient
        input_gradient = np.dot(self.W.T, output_gradient)
        W_gradient = np.dot(output_gradient , self.X.T)
        self.W -= learning_rate * W_gradient
        self.B -= learning_rate * output_gradient
        return input_gradient