import numpy as np

from utils import compute_entropy
from collections import Counter

class Node():
    
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    
    def __init__(self,min_samples_split=2,max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        #greedy search
        best_feature, best_thresh = self._best_criteria(X,y,feat_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feature], best_thresh)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature,best_thresh,left,right)
    
    def _most_common_label(self,y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _best_criteria(self, X, y, feat_idxs):
        larger_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > larger_gain:
                    larger_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
    
        return split_idx, split_thresh
    
    def _information_gain(self,y, X_column, split_thresh):
        #parent entropy
        parent_entropy = compute_entropy(y)
        
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        #weighted average child entropy
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = compute_entropy(y[left_idxs]) , compute_entropy(y[right_idxs])
        child_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
        
        #return ig
        return parent_entropy - child_entropy
    
    
    def _split(self,X_column, split_tresh):
        left_idxs = np.argwhere(X_column <= split_tresh).flatten()
        right_idxs = np.argwhere(X_column > split_tresh).flatten()
        return left_idxs, right_idxs
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
        
    def predict(self, X):
        
        return np.array([self._traverse_tree(x,self.root) for x in X])