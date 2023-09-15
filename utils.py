import numpy as np
import copy
import math

def eucledian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def mse_cost(X,y,W,B):
    m = X.shape[0]
    f_wb = np.dot(X,W) + B
    return ((f_wb - y)**2) / (2*m)

def compute_linear_gradient(X, y, W, B):
    f_wb = np.dot(X,W) + B
    dj_dw = (1 / X.shape[0]) * np.dot(X.T, (f_wb - y))
    dj_db = (1 / X.shape[0]) * np.sum(f_wb - y)
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, verbose):
    m = X.shape[0]
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 10000: # prevent resource exhaustion
            cost = cost_function(X,y,w,b)
            J_history.append(cost)
        if (verbose and i % math.ceil(num_iters / 10) == 0 or i == num_iters - 1):
            print(f"Iteration {i} - Cost:{float(J_history[-1][-1]):8.2f}")
    return w, b, J_history