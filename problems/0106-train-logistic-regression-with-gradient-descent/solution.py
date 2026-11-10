import numpy as np
def sigmoid(x: float) -> float:
    return 1/(1+np.exp(-x))
def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    w = np.zeros((X.shape[1]+1, 1))
    vector_one = np.ones((X.shape[0], 1))
    X = np.hstack((vector_one, X))
    loss = []
    for i in range(iterations):
        loss_element = -sum([ y[j] * np.log(sigmoid(X[j]@w)) + (1 - y[j]) * np.log(1 - sigmoid(X[j]@w)) for j in range(X.shape[0]) ])
        loss.append(loss_element)
        y_sigmoid = np.asarray([sigmoid(X[i]@w) for i in range(X.shape[0])])
        w = w - learning_rate * ( X.T@(y_sigmoid - y.reshape(-1, 1)) )
    loss = np.asarray(loss)
    return (np.round(w.flatten(), 4), np.round(loss.flatten(), 4))


