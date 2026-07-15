# vectorizer_tfidf.py
import numpy as np
import math
from collections import Counter

class TFIDFVectorizerScratch:
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df
        self.vocab = {}
        self.idf = None

    def fit(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(d))
        N = len(docs)

        vocab = {}
        for term, freq in df.items():
            if freq >= self.min_df and freq/N <= self.max_df:
                vocab[term] = len(vocab)

        self.vocab = vocab
        self.idf = np.zeros(len(vocab))

        for t, i in vocab.items():
            self.idf[i] = math.log((N+1)/(df[t]+1)) + 1.0
        return self

    def transform(self, docs):
        M = len(docs)
        V = len(self.vocab)
        mat = np.zeros((M, V))
        for i, d in enumerate(docs):
            tf = Counter(d)
            for t, c in tf.items():
                if t in self.vocab:
                    j = self.vocab[t]
                    mat[i,j] = (c / len(d)) * self.idf[j]
        return mat

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)
