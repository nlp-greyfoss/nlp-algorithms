import torch
import torch.nn as nn

from transformer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    '''
    BERT的嵌入由三个嵌入层组成：
    1. 标记嵌入层(TokenEmbedding)
    2. 位置嵌入层(PositionalEmbedding)
    3. 片段嵌入层(SegmentEmbedding)
    BERT输入中的每个单词分别输入到三个嵌入层中，得到三个嵌入向量表示，然后这三个表示加起来就是BERT的输入嵌入。
    这三个层都是通过学习得来的。
    '''

    def __init__(self, vocab_size, d_embed, type_vocab_size, max_position_embeddings, dropout=0.1):
        '''

        :param vocab_size: 词表大小
        :param d_embed: 嵌入层大小
        :param type_vocab_size: 类型词表大小
        :param max_position_embeddings: 最大输入长度
        :param dropout: dropout比率，通常为0.1
        '''

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_embed)
        self.segment_embedding = nn.Embedding(type_vocab_size, d_embed)

        self.layer_norm = LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        '''

        :param input_ids: 输入标记的ID列表
        :param token_type_ids: 片段ID，如果是句子对(A,B)，比如来自A句子的单词映射为0，来自B句子的单词映射为1
        :return:
        '''
        seq_len = input_ids.size(1)
        # position_ids就是位置信息[0, 1, 2, ..., seq_len - 1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (seq_len, ) -> (batch_size, seq_len)
        # 如果输入的是单个句子，则全部为0即可
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
