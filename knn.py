"""
used general case eucledian distance 
"""
import numpy as np
from collections import Counter
import utils as ut

class KNN:
    
    def __init__(self,k=3):
       self.k = k
    
    def fit(self,X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self,x):
        #compute the distances
        distances = [ut.eucledian_distance(x, x_train) for x_train in self.X_train]
        
        #get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        K_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #majority vote, most common label   
        #most common method outputs a list of tuples, the first index of the tuple is the item and the second is the number of times it appears
        #therefore the use of ().most_common(1)[0][0]
        most_common = Counter(K_nearest_labels).most_common(1)[0][0] 
        
        return most_common
        