# nb_bernoulli.py
import numpy as np
import math

class BernoulliNBScratch:
    def __init__(self, alpha=1e-2):
        self.alpha = alpha

    def binarize(self, X):
        return (X > 0).astype(int)

    def fit(self, X, y):
        X = self.binarize(X)
        N, V = X.shape
        self.classes = np.unique(y)
        self.log_prior = {}
        self.log_prob1 = {}
        self.log_prob0 = {}

        for c in self.classes:
            Xc = X[y==c]
            Nc = Xc.shape[0]
            self.log_prior[c] = math.log(Nc / N)

            p = (Xc.sum(axis=0) + self.alpha) / (Nc + 2*self.alpha)
            self.log_prob1[c] = np.log(p)
            self.log_prob0[c] = np.log(1-p)

        return self

    def predict(self, X):
        X = self.binarize(X)
        N = X.shape[0]
        scores = np.zeros((N, len(self.classes)))
        for idx, c in enumerate(self.classes):
            scores[:, idx] = self.log_prior[c] + \
                             X @ self.log_prob1[c] + \
                             (1-X) @ self.log_prob0[c]
        return self.classes[np.argmax(scores, axis=1)]
