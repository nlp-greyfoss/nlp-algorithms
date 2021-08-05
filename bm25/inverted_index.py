import collections


class Dictionary(dict):
    '''
    data structure for InvertedIndex
    word -> postings(doc_id->word_count)
    '''

    def __missing__(self, key):
        postings = collections.defaultdict(int)
        self[key] = postings
        return postings


class InvertedIndex:
    def __init__(self):
        self.dictionary = Dictionary()

    def add(self, doc_id, doc):
        for word in doc:
            postings = self.dictionary[word]
            postings[doc_id] += 1

    def __contains__(self, word):
        return word in self.dictionary

    def __getitem__(self, word):
        return self.dictionary[word]

    def __len__(self):
        '''
        :return: word count in corpus
        '''
        return len(self.dictionary)

    def get_doc_frequency(self, word):
        '''
        get number of docs word occurs in
        :param word:
        :return:
        '''
        return len(self.dictionary[word])
