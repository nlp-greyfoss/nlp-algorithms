import torch.nn as nn

from bert.layers.token_embedding import TokenEmbedding


class BERTEmbedding(nn.Module):
    '''
    BERT的嵌入由三个嵌入层组成：
    1. 标记嵌入层 TokenEmbedding
    2. 位置嵌入层 PositionalEmbedding
    3. 片段嵌入层 SegmentEmbedding
    BERT输入中的每个单词分别输入到三个嵌入层中，得到三个嵌入向量表示，然后这三个表示加起来就是BERT的输入嵌入
    '''

    def __init__(self, d_vocab, d_embed, dropout=0.1):
        '''
        :param d_vocab: 词表大小
        :param d_embed: 嵌入层大小
        :param dropout: Dropout比率
        '''
        super.__init__()
        self.token = TokenEmbedding()
