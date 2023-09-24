import numpy as np
from decision_tree import DecisionTree
from utils import bootstrap_sample, most_common_label


class RandomForest:
    
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, n_feats=None, feature_subsamp=False):
        self.n_trees = n_trees
        self.min_samples_split =min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.feature_subsamp = feature_subsamp
        self.trees = []
        
    def fit(self,X,y):
        self.trees = []
        self.feature_idxs = None
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth,n_features=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X,y)
            
            if self.feature_subsamp:
                max_features = int(np.sqrt(X.shape[1]))
                self.feature_idxs = np.random.choice(X.shape[1], size= max_features, replace=False)
                X_sample = X_sample[:, self.feature_idxs]
                
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_predictions = None
        if self.feature_subsamp:
            tree_predictions = [tree.predict(X[:, self.feature_idxs]) for tree in self.trees]
        else:
            tree_predictions = [tree.predict(X) for tree in self.trees]
        
        # Aggregate predictions from individual trees (e.g., by majority vote)
        y_pred = np.apply_along_axis(lambda x: most_common_label(x), axis=0, arr=np.array(tree_predictions))
        return y_pred
    