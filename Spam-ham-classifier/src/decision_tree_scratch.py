# src/decision_tree_scratch.py
import numpy as np
import math
import random

class TreeNode:
    def __init__(self, is_leaf=False, pred=None, feat=None, thr=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.pred = pred
        self.feat = feat
        self.thr = thr
        self.left = left
        self.right = right

def entropy(y):
    if len(y) == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1.0 - p1
    ent = 0.0
    if p0 > 0:
        ent -= p0 * math.log2(p0)
    if p1 > 0:
        ent -= p1 * math.log2(p1)
    return ent

class DecisionTreeEntropy:
    """
    Binary decision tree using Entropy (Information Gain).
    - max_depth: stops recursion at this depth
    - min_samples_split: minimum samples required to attempt split
    - max_features: number of features to sample at each split (None => all)
    - n_thresholds: number of thresholds per feature (chosen from quantiles) to evaluate
    """
    def __init__(self, max_depth=12, min_samples_split=2, max_features=None, n_thresholds=10, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.max_features = max_features
        self.n_thresholds = max(2, n_thresholds)
        self.root = None
        self.random_state = None if random_state is None else int(random_state)
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        N, V = X.shape
        base_ent = entropy(y)
        best = {'feat': None, 'thr': None, 'gain': 0.0}
        # choose features to consider
        features = list(range(V))
        if self.max_features is not None and self.max_features < V:
            features = list(np.random.choice(V, min(self.max_features, V), replace=False))
        for j in features:
            col = X[:, j]
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue
            # choose candidate thresholds as quantiles to limit computation
            qs = np.linspace(0.05, 0.95, min(self.n_thresholds, uniq.size-1))
            thresholds = np.unique(np.quantile(uniq, qs))
            for thr in thresholds:
                left_mask = col <= thr
                nl = left_mask.sum()
                nr = N - nl
                if nl < self.min_samples_split or nr < self.min_samples_split:
                    continue
                ent_l = entropy(y[left_mask])
                ent_r = entropy(y[~left_mask])
                w_l = nl / N
                w_r = nr / N
                gain = base_ent - (w_l * ent_l + w_r * ent_r)
                if gain > best['gain']:
                    best = {'feat': j, 'thr': float(thr), 'gain': float(gain)}
        return best if best['feat'] is not None else None

    def _build(self, X, y, depth):
        # stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.unique(y).size == 1:
            pred = 1 if np.mean(y) >= 0.5 else 0
            return TreeNode(is_leaf=True, pred=pred)
        split = self._best_split(X, y)
        if split is None or split['gain'] <= 0.0:
            pred = 1 if np.mean(y) >= 0.5 else 0
            return TreeNode(is_leaf=True, pred=pred)
        feat = split['feat']
        thr = split['thr']
        left_idx = X[:, feat] <= thr
        right_idx = ~left_idx
        left = self._build(X[left_idx], y[left_idx], depth+1)
        right = self._build(X[right_idx], y[right_idx], depth+1)
        return TreeNode(is_leaf=False, feat=feat, thr=thr, left=left, right=right)

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.pred
        if x[node.feat] <= node.thr:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        if self.root is None:
            raise RuntimeError("The tree has not been trained. Call fit(X,y) first.")
        preds = np.array([self._predict_one(row, self.root) for row in X], dtype=int)
        return preds
