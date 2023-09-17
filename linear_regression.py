import numpy as np
import utils as ut

class LinearRegression:
    
    def __init__(self,w=None,b=None):
        self.w = w
        self.b = b
    
    def fit(self, X, y,learning_rate=0.001, num_iters=1000,verbose=True,lambda_=0):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.w, self.b, _ = ut.gradient_descent(X,y,self.w,self.b,ut.mse_cost,ut.compute_gradient,learning_rate,num_iters,verbose,lambda_)
        
    def predict(self,X):
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained, please call the fit() method first")
        return np.dot(X, self.w) + self.b    