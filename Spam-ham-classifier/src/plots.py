# plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_roc(y_true, y_scores, path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    A = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={A:.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(path)
    plt.close()

def plot_tfidf_hist(X, path):
    vals = X.flatten()
    vals = vals[vals > 0]   # only non-zero TF-IDF values

    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=40, color="steelblue", edgecolor="black")

    plt.xlabel("TF-IDF value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Non-Zero TF-IDF Features")
    plt.tight_layout()

    plt.savefig(path, dpi=300)
    plt.close()
