import pandas as pd
import jieba
from bm25.okapi_bm25 import OkapiBM25


def main():
    data_path = './data/questions.csv'
    df = pd.read_csv(data_path, usecols=['content'])
    corpus = df['content'].tolist()

    bm25 = OkapiBM25(corpus, tokenize=jieba.lcut)

    query = "怀孕吃什么"

    result = bm25.get_top_k(query, corpus=corpus, k=5)

    for q in result:
        print(q)


if __name__ == '__main__':
    main()
