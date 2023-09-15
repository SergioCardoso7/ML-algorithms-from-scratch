import numpy as np
import utils as ut

class logisticRegression:

    def __init__(self,w=None,b=None):
        self.w = w
        self.b = b
    
    def fit(self,X,y,learning_rate=0.001,num_iters=1000,verbose=True):
        
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        self.w, self.b, J_history = ut.gradient_descent(X,y,self.w,self.b,ut.logistic_cost,ut.compute_gradient,learning_rate,num_iters,verbose,logistic=True)
    
    def predict(self,x,threshold):
        if self.w is None or self.b is None:
            raise ValueError("Model has not been trained, please call the fit() method first")
        
        return ut.sigmoid(np.dot(x,self.w) + self.b) >= threshold
        
        