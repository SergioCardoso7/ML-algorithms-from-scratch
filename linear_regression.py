import numpy as np
import utils as ut
"""
using mean squared error cost function
"""
class LinearRegression:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y,learning_rate=0.001, num_iters=1000,verbose=True):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
            
        self.w, self.b, J_history = ut.gradient_descent(X,y,self.w,self.b,ut.mse_cost,ut.compute_gradient,learning_rate,num_iters,verbose)
        
    def predict(self,X):
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained, please call the fit() method first")
        
        return np.dot(X, self.w) + self.b    
   