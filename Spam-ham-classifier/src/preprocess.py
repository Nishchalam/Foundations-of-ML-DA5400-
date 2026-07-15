# preprocess.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
import glob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    tokens = [t for t in tokens if t not in STOP]
    return tokens

def load_labeled(folder):
    X, y = [], []
    for cls, label in [("Ham",0),("Spam",1)]:
        for f in glob.glob(os.path.join(folder, cls, "*.txt")):
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                X.append(clean_text(fh.read()))
                y.append(label)
    return X, y

def load_test(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    out = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            out.append(clean_text(fh.read()))
    return files, out
