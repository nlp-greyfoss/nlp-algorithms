import torch.nn as nn

from bert.layers.bert_embedding import BertEmbedding


class BERT(nn.Module):
    '''
    BERT 模型实现
    '''

    def __init__(self, d_vocab, d_model=768, n_layers=12, attn_heads=12, dropout=0.1, d_type_vocab=2,
                 max_position_embeddings=512):
        '''

        :param d_vocab: 词表大小
        :param d_model: 隐藏层大小，即BERT模型的输出大小
        :param n_layers: 编码器层数
        :param attn_heads: 多头注意力头数
        :param dropout: dropout比率，通常为0.1
        :param d_type_vocab: 类型词表大小
        :param max_position_embeddings: 最大输入长度
        '''
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # 前馈神经网络的隐藏层大小
        self.d_ff = d_model * 4

        self.embedding = BertEmbedding(d_vocab, d_model, d_type_vocab, max_position_embeddings, dropout)
        self.transformer