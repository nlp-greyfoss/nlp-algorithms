import unittest
from bm25.okapi_bm25 import *


class UnitTest(unittest.TestCase):

    def test_inverted_index(self):
        idx = InvertedIndex()

        corpus = [['one', 'two', 'three'], ['一', '二', '三'], ['1', '2', '3'], ['one', '1', '一'],
                  'in a big big world'.split()]

        for doc_id, doc in enumerate(corpus):
            idx.add(doc_id, doc)

        self.assertEqual(idx.get_doc_frequency('one'), 2)
        self.assertEqual(idx.get_doc_frequency('2'), 1)
        self.assertTrue('A' not in idx)
        self.assertTrue('1' in idx)
        self.assertEqual(len(idx['one']), 2)
        self.assertEqual(idx['one'], {0: 1, 3: 1})

    def test_doc_len(self):
        dl = DocLen()
        dl.put(0, 11)
        dl.put(1, 23)
        dl.put(2, 34)

        self.assertEqual(dl[0], 11)
        self.assertEqual(dl[4], 0)
        self.assertAlmostEqual(dl.get_avg_len(), 22.67, delta=0.05)
        self.assertEqual(len(dl), 3)

    def test_okapi_bm25(self):
        corpus = [
            "你好 你 叫 什么",
            "你 的 名字 是 什么",
            "你 今天 吃了 没"
        ]

        bm25 = OkapiBM25(corpus, tokenize=lambda x: x.split())
        query = "你 叫 什么 名字"

        doc_scores = bm25.get_scores(query)
        print(doc_scores)

        self.assertEqual(bm25.get_top_k(query, k=1), [0])


if __name__ == '__main__':
    unittest.main()
