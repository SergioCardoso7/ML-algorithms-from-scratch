import numpy as np

class PCA:
    
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self,X):
       
        #covariance
        cov = np.cov(X.T)
        #eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #sort eigenvectors in decreasing order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues[::-1])
        eigenvalues = eigenvalues[idxs]
        eigenvectors =eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]
    
    def transform(self,X):
        #project data
        
        return np.dot(X, self.components.T)