import numpy as np
import copy
from math import ceil

def eucledian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def sigmoid(z):
    return 1 / (1+ np.exp(-z))

def mean_norm(X):
    return (X - np.mean(X)) / (np.max(X) - np.min(X))

def z_score_norm(X):
    return (X - np.mean(X)) / np.std(X)

def mse_loss(y_true, y_pred):
    return 0.5 * np.mean(np.square(y_pred - y_true))

def mse_loss_prime(y_true, y_pred):
    return (y_pred - y_true) / np.size(y_true)

def mse_cost(X,y,W,B,lambda_=0):
    f_wb = np.dot(X,W) + B
    return (1/2) * np.mean(((f_wb - y)**2)) + (lambda_ / (2*X.shape[0])) * np.sum(np.square(W))

def logistic_cost(X,y,W,B, lambda_ = 0,epsilon=1e-15):
    f_wb = sigmoid(np.dot(X, W) + B)
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon) # Apply epsilon smoothing to avoid taking the logarithm of zero or small values a.k.a avoid log explosion
    return np.mean(-y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)) + (lambda_ / (2*X.shape[0])) * np.sum(np.square(W))

def compute_gradient(X, y, W, B, logistic=False, lambda_=0):
    f_wb = np.dot(X,W) + B
    if logistic:
        f_wb = sigmoid(f_wb)
    dj_dw = (1 / X.shape[0]) * np.dot(X.T, (f_wb - y)) + (lambda_ / X.shape[0]) * W
    dj_db = (1 / X.shape[0]) * np.sum(f_wb - y)  
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, verbose,lambda_=0,logistic=False):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X,y,w,b,logistic,lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if (verbose and i % ceil(num_iters / 10) == 0) or (verbose and i == num_iters - 1):
            J_history.append(cost_function(X,y,w,b,lambda_))
            print(f"Iteration {i} - Cost:{float(J_history[-1]):8.2f}")
    return w, b, J_history

def unit_step_func(X):
    return np.where(X >=0 , 1, 0)
