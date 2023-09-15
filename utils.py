import numpy as np
import copy
import math

def eucledian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def sigmoid(z):
    return 1 / (1+ np.exp(-z))

def mse_cost(X,y,W,B):
    m = X.shape[0]
    f_wb = np.dot(X,W) + B
    return ((f_wb - y)**2) / (2*m)

def logistic_cost(X,y,W,B, epsilon=1e-15):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, W) + B)
    
    # Apply epsilon smoothing to avoid taking the logarithm of zero or small values
    # avoid log explosion
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    
    cost = (-y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)).mean()
    return cost

def compute_gradient(X, y, W, B, logistic=False):
    f_wb = np.dot(X,W) + B
    if logistic:
        f_wb = sigmoid(f_wb)
    dj_dw = (1 / X.shape[0]) * np.dot(X.T, (f_wb - y))
    dj_db = (1 / X.shape[0]) * np.sum(f_wb - y)  
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, verbose,logistic=False):
    m = X.shape[0]
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X,y,w,b,logistic)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 10000: # prevent resource exhaustion
            cost = cost_function(X,y,w,b)
            J_history.append(cost)
        if (verbose and i % math.ceil(num_iters / 10) == 0) or (verbose and i == num_iters - 1):
            print(f"Iteration {i} - Cost:{float(J_history[-1]):8.2f}")
    return w, b, J_history


