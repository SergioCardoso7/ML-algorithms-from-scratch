"""
P(y|X) = (P(X|y) * P(y)) / P(X)

Naive Bayes assumes that all features are mutually independent

P(y|X) = ( P(X1|y) * P(X2|y) * P(X3|y) * P(X4|y) *...* P(y) ) / P(X)

Select class with highest probability - Multinomial Naive Bayes Classifier

log function is used to prevent overflow and also the case of one or more of the probabilities being zero

y = argmaxy( log(P(X1|y)) + log(P(X2|y)) + log(P(X3|y)) + log(P(X4|y)) + ... + log(P(y))  ) -> case of one class

Prior probability P(y) : is the frequency

class conditional probability (of one class) is modeled with the gaussian distribution
"""

import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes,n_features) ,dtype=np.float64)
        self._var = np.zeros((n_classes,n_features) ,dtype=np.float64)
        self._priors = np.zeros(n_classes,dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[idx,:] = X_c.mean(axis=0)    
            self._var[idx,:] = X_c.var(axis=0)    
            self._priors[idx] = X_c.shape[0] / float(n_samples)
                     
    def predict(self,X):
        return [self._predict(x) for x in X]
    
    def _predict(self,x):
        posteriors = []
        
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditionals = np.sum(np.log(self._prob_dense_func(idx,x)))
            posterior = prior + class_conditionals
            posteriors.append(posterior)        
        return self._classes[np.argmax(posteriors)]
    
    def _prob_dense_func(self, class_idx,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator / denominator