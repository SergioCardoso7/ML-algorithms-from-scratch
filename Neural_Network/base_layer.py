import numpy as np

class BaseLayer:
    
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input):
        # returns output
        pass
    
    def backward(self,output_gradient, learning_rate):
        # updates trainable parameters and returns input gradient
        pass