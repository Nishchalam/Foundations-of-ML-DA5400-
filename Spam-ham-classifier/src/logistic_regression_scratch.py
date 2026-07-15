# logistic_regression_scratch.py
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=300):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        N, V = X.shape
        self.w = np.zeros(V)
        self.b = 0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self.sigmoid(z)
            grad_w = (1/N) * (X.T @ (p - y))
            grad_b = (1/N) * np.sum(p - y)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X):
        return (self.sigmoid(X @ self.w + self.b) >= 0.5).astype(int)
