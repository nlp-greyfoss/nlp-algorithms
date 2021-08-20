import numpy as np
import math

from bm25.doc_len import DocLen
from bm25.inverted_index import InvertedIndex
import re

class OkapiBM25:
    def __init__(self, corpus, k=1.2, b=.75, tokenize=None):

        self.tokenize = tokenize

        self.k = k
        self.b = b

        if tokenize:
            corpus = [tokenize(doc) for doc in corpus]

        self.idx, self.dl = self.build_struct(corpus)


    @staticmethod
    def build_struct(corpus):
        idx = InvertedIndex()
        dl = DocLen()
        for doc_id, doc in enumerate(corpus):
            idx.add(doc_id, doc)
            dl.put(doc_id, len(doc))

        return idx, dl

    def get_score(self, word, doc_id, doc_freq):
        '''
        compute Okapi bm25
        :param word:
        :param doc_id: document index in corpus
        :param doc_freq: number of docs word occurs in
        :return:
        '''
        tf = math.log10(self.idx[word][doc_id] + 1)

        if not tf:
            return 0

        tf = self._compute_weighted_tf(tf, self.dl[doc_id], self.dl.get_avg_len())
        idf = self._compute_probabilistic_idf(doc_freq)
        return tf * idf

    def _compute_weighted_tf(self, tf, doc_len, avg_doc_len):
        return tf * (self.k + 1) / (self.k * (1 - self.b + self.b * doc_len / avg_doc_len) + tf)

    def _compute_probabilistic_idf(self, df):
        return math.log((len(self.dl) + 1) / (df + 0.5))

    def get_scores(self, query):
        query = self.tokenize(query) if self.tokenize else query

        scores = np.zeros(len(self.dl))

        for word in query:
            if word in self.idx:
                for doc_id in self.idx[word].keys():
                    scores[doc_id] += self.get_score(word, doc_id, self.idx.get_doc_frequency(word))

        return scores

    def get_top_k(self, query, corpus=None, k=10):
        '''
        return top k documents
        :param query:
        :param corpus:
        :param k:
        :return:
        '''
        scores = self.get_scores(query)
        top_k = np.argsort(scores)[::-1][:k]
        if corpus:
            assert len(corpus) == len(self.dl)
            return [corpus[i] for i in top_k]
        return top_k
