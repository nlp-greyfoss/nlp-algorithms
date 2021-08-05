class DocLen:
    '''
    Save all document length in corpus

    doc_id -> doc_len
    '''

    def __init__(self):
        self.avg_len = 0.0
        self.dict = dict()

    def put(self, doc_id, doc_len):
        self.dict[doc_id] = doc_len

    def __getitem__(self, doc_id):
        return self.dict.get(doc_id, 0)

    def __len__(self):
        '''
        :return: document count in corpus
        '''
        return len(self.dict)

    def get_avg_len(self):
        '''
        get average document length
        :return:
        '''
        if not self.avg_len:
            total_count = sum(self.dict.values())
            self.avg_len = total_count / len(self.dict)

        return self.avg_len
