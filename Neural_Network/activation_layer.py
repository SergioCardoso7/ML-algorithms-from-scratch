from base_layer import BaseLayer
import numpy as np

class Activation(BaseLayer):
    
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime 
        
    def forward(self, X):
        self.X = X
        return self.activation(self.X)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.X))