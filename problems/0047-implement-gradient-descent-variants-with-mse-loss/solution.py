import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    w = np.zeros((X.shape[1], 1))
    y = y.reshape(-1, 1)
    len = X.shape[0]
    if(method == 'batch'):
        for i in range(n_epochs):
            w = w - 2*(learning_rate / len) * (X.T @ (X@w - y))
        return w.flatten()
    elif(method == 'mini_batch'):
        for i in range(n_epochs):
            j = 0
            while(j<len):
                X_mini = X[j : j+batch_size]
                y_mini = y[j : j+batch_size]
                w = w - 2 * (learning_rate / batch_size) * (X_mini.T @ (X_mini @ w - y_mini))
                j = j + batch_size
        return w.flatten()
    else:
        for i in range(n_epochs):
            for j in range(len):
                w = w - 2 * learning_rate * (X[j%len].reshape(1, -1).T @ (X[j%len].reshape(1,-1) @ w - y[j%len].reshape(-1,1)))
        return w.flatten()
