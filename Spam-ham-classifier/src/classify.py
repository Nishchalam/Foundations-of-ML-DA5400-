# classify.py
from preprocess import load_test
import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

OUT = "models"
model_name = "NB"     
test_dir = "data/test"  

if not os.path.isdir(test_dir) and os.path.isdir("test"):
    test_dir = "test"

model_path = os.path.join(OUT, f"{model_name}.pkl")
vectorizer_path = os.path.join(OUT, "vectorizer.pkl")
output_pred_path = os.path.join(OUT, f"output_predictions_{model_name}.txt")
metrics_out_path = os.path.join(OUT, f"metrics_{model_name}.txt")
confmat_out_path = os.path.join(OUT, f"confmat_{model_name}.png")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
if not os.path.isfile(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

with open(model_path, "rb") as fh:
    model = pickle.load(fh)

with open(vectorizer_path, "rb") as fh:
    vec = pickle.load(fh)

files, docs = load_test(test_dir)
if len(files) == 0:
    print(f"No test files found in {test_dir}. Create test/*.txt or point test_dir to correct folder.")
    raise SystemExit(1)

X = vec.transform(docs)

try:
    X_arr = np.asarray(X)
except Exception:
    X_arr = X

preds = model.predict(X_arr)

# write predictions
with open(output_pred_path, "w", encoding="utf-8") as f:
    for fn, p in zip(files, preds):
        f.write(f"{os.path.basename(fn)}\t{'+1' if int(p) == 1 else '0'}\n")

print(f"Predictions written to: {output_pred_path}")


labels_csv_candidates = [
    os.path.join(test_dir, "test_labels.csv"),
    os.path.join(test_dir, "labels.csv"),
    os.path.join(test_dir, "test_labels.txt"),
]

labels_path = None
for c in labels_csv_candidates:
    if os.path.isfile(c):
        labels_path = c
        break

if labels_path is None:
    print("No test label file found (expected one of: test_labels.csv / labels.csv in test folder).")
    print("Skipping accuracy computation. To compute accuracy, provide a CSV with columns 'filename,label' where label is '+1' or '0'.")
else:
    # read labels
    df = pd.read_csv(labels_path, dtype=str)
    # accept both headers or simple two-column no-header
    if 'filename' in df.columns.str.lower() or 'label' in df.columns.str.lower():
        # normalize column names case-insensitively
        cols = {c.lower(): c for c in df.columns}
        fn_col = cols.get('filename') or df.columns[0]
        lab_col = cols.get('label') or df.columns[1]
    else:
        # assume first column filename, second label
        fn_col = df.columns[0]
        lab_col = df.columns[1]
    # build mapping filename -> label (0 or 1)
    mapping = {}
    for _, row in df.iterrows():
        fname = os.path.basename(str(row[fn_col])).strip()
        raw = str(row[lab_col]).strip()
        if raw == '+1' or raw == '1':
            mapping[fname] = 1
        else:
            try:
                mapping[fname] = int(raw)
            except:
                mapping[fname] = 1 if raw.startswith('+') else 0

    # align labels with files order
    y_true = []
    missing = []
    for fn in files:
        b = os.path.basename(fn)
        if b in mapping:
            y_true.append(mapping[b])
        else:
            y_true.append(0)   # default fallback to ham if missing
            missing.append(b)

    if missing:
        print("Warning: the following test files had no label in the CSV; assumed 0 (ham):")
        for m in missing:
            print("  ", m)

    # compute metrics
    metrics = compute_metrics(np.array(y_true), np.array(preds))
    # print metrics
    print("Evaluation metrics on test set:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print("  Confusion matrix counts: TP, TN, FP, FN =", metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'])

    # save metrics to file
    with open(metrics_out_path, "w", encoding="utf-8") as fh:
        fh.write("Evaluation metrics on test set\n")
        fh.write(f"Model: {model_name}\n")
        fh.write(f"Test folder: {test_dir}\n")
        fh.write(f"Label file used: {labels_path}\n\n")
        fh.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        fh.write(f"Precision: {metrics['precision']:.6f}\n")
        fh.write(f"Recall: {metrics['recall']:.6f}\n")
        fh.write(f"F1: {metrics['f1']:.6f}\n")
        fh.write(f"TP: {metrics['tp']}\nTN: {metrics['tn']}\nFP: {metrics['fp']}\nFN: {metrics['fn']}\n")

    print(f"Metrics written to: {metrics_out_path}")

    cm = np.array([
    [metrics["tn"], metrics["fp"]],
    [metrics["fn"], metrics["tp"]],])

    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center', fontsize=13)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(confmat_out_path)
    plt.close()

    print(f"Confusion matrix image saved to: {confmat_out_path}")
