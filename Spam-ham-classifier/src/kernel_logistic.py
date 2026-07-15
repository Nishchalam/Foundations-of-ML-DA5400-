import numpy as np

def linear_kernel(X, Y):
    return X @ Y.T

class KernelLogisticRegression:
    """
    Kernelized Logistic Regression (dual) using gradient descent on alpha:
      f = K @ alpha
    minimize  L = sum log(1 + exp(-y * f)) + (lambda/2) alpha^T K alpha
    where y in {-1,+1}, K is Gram matrix (train x train).
    predict_proba for test via sigmoid(K_test @ alpha).
    """
    def __init__(self, kernel=None, lam=1e-3, lr=0.1, max_iter=500, tol=1e-6, verbose=False):
        self.kernel = kernel if kernel is not None else linear_kernel
        self.lam = float(lam)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.alpha = None
        self.X = None
        self.y = None
        self.K = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y = np.where(y == 0, -1, y)
        N = X.shape[0]
        self.X = X
        self.y = y
        # Gram matrix
        K = self.kernel(X, X)
        self.K = K
        # initialize alpha
        alpha = np.zeros(N, dtype=float)
        # gradient descent on alpha
        for it in range(self.max_iter):
            f = K @ alpha                     # shape (N,)
            y_f = y * f
            # gradient of logistic loss w.r.t f is -y * sigmoid(-y*f)
            sigma = 1.0 / (1.0 + np.exp(y_f))  # sigmoid(-y*f)
            grad_f = - y * sigma               # dL/df per sample
            # chain rule: dL/dalpha = K @ grad_f  + lambda * K @ alpha
            grad_alpha = K @ grad_f + self.lam * (K @ alpha)
            # step
            alpha_new = alpha - self.lr * grad_alpha
            diff = np.linalg.norm(alpha_new - alpha)
            alpha = alpha_new
            if self.verbose and (it % 50 == 0 or diff < self.tol):
                # compute current loss
                loss = np.mean(np.log(1.0 + np.exp(-y_f))) + 0.5 * self.lam * alpha.T @ K @ alpha
                print(f"iter {it:04d} loss={loss:.6f} diff={diff:.6e}")
            if diff < self.tol:
                break
        self.alpha = alpha
        return self

    def decision_function(self, X_test):
        X_test = np.asarray(X_test)
        K_test = self.kernel(X_test, self.X)   # shape (n_test, N_train)
        scores = K_test @ self.alpha
        return scores

    def predict_proba(self, X_test):
        scores = self.decision_function(X_test)
        # probability using sigmoid of scores; convert to (n,2) like sklearn
        p1 = self._sigmoid(scores)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X_test):
        p = self.predict_proba(X_test)[:,1]
        return (p >= 0.5).astype(int)
