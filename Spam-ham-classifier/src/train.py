# train.py
import os
import pickle
import numpy as np

from preprocess import load_labeled
from vectorizer_tfidf import TFIDFVectorizerScratch
from nb_bernoulli import BernoulliNBScratch
from logistic_regression_scratch import LogisticRegressionScratch
from kernel_logistic import KernelLogisticRegression
from decision_tree_scratch import DecisionTreeEntropy

from plots import plot_cm, plot_roc, plot_tfidf_hist

import numpy as np

class RBFKernel:
    def __init__(self, gamma=0.5):
        self.gamma = float(gamma)

    def __call__(self, A, B):
        # expects A: (nA, d), B: (nB, d)
        A = np.asarray(A)
        B = np.asarray(B)
        A_sq = np.sum(A * A, axis=1)[:, None]   # (nA,1)
        B_sq = np.sum(B * B, axis=1)[None, :]   # (1,nB)
        # squared distance matrix
        sq = A_sq + B_sq - 2.0 * (A @ B.T)
        return np.exp(-self.gamma * sq)

# ----------------------------
# Paths
# ----------------------------
DATA_TRAIN = "data/Train"
DATA_VAL = "data/TrainAccuracy"
OUT = "models"

os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/figures", exist_ok=True)

# ----------------------------
# Load training + validation
# ----------------------------
Xtr_raw, ytr = load_labeled(DATA_TRAIN)
Xval_raw, yval = load_labeled(DATA_VAL)

# ----------------------------
# TF-IDF feature extraction
# ----------------------------
vec = TFIDFVectorizerScratch(min_df=2, max_df=0.95)
Xtr = vec.fit_transform(Xtr_raw)
Xval = vec.transform(Xval_raw)

# ensure numpy arrays
Xtr = np.asarray(Xtr)
Xval = np.asarray(Xval)
ytr = np.asarray(ytr)
yval = np.asarray(yval)

# plot distribution
plot_tfidf_hist(Xtr, f"{OUT}/figures/tfidf_hist.png")

# ----------------------------
# Define kernel functions
# ----------------------------
def rbf_kernel_factory(gamma):
    def k(A, B):
        A_sq = np.sum(A*A, axis=1)[:,None]
        B_sq = np.sum(B*B, axis=1)[None,:]
        sq = A_sq + B_sq - 2*(A @ B.T)
        return np.exp(-gamma * sq)
    return k

# ----------------------------
# Models to train
# ----------------------------
models = {
    "NB": BernoulliNBScratch(alpha=1e-2),
    "LR": LogisticRegressionScratch(lr=0.01, epochs=200),
    "KLR_RBF": KernelLogisticRegression(kernel=RBFKernel(gamma=0.5),lam=1e-3, lr=0.5, max_iter=500),
    "DT": DecisionTreeEntropy(max_depth=12, min_samples_split=4, max_features=50),
}

# ----------------------------
# Training loop
# ----------------------------
results = []

for name, clf in models.items():
    print(f"\nTraining {name}...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xval)

    # ---- ROC plots ----
    if name == "LR":
        scores = Xval @ clf.w + clf.b
        plot_roc(yval, scores, f"{OUT}/figures/roc_{name}.png")

    if name == "KLR_RBF":
        scores = clf.predict_proba(Xval)[:,1]
        plot_roc(yval, scores, f"{OUT}/figures/roc_{name}.png")

    # ---- Confusion Matrix ----
    plot_cm(yval, pred, f"{OUT}/figures/cm_{name}.png")

    # ---- Record Accuracy ----
    acc = np.mean(pred == yval)
    results.append((name, acc))
    print(f"{name} accuracy = {acc:.4f}")

    # ---- Save Model ----
    with open(f"{OUT}/{name}.pkl", "wb") as fh:
        pickle.dump(clf, fh)

# ----------------------------
# Save vectorizer
# ----------------------------
with open(f"{OUT}/vectorizer.pkl","wb") as fh:
    pickle.dump(vec, fh)

print("\nResults:", results)
