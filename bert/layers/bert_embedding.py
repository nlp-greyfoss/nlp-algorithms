import torch
import torch.nn as nn

from bert.layers.layer_norm import BertLayerNorm
from transformer.layers.layer_norm import LayerNorm


class BertEmbeddings(nn.Module):
    '''
    BERT的嵌入由三个嵌入层组成：
    1. 标记嵌入层(WordEmbedding)
    2. 位置嵌入层(PositionalEmbedding)
    3. 片段嵌入层(TokenTypeEmbedding)
    BERT输入中的每个单词分别输入到三个嵌入层中，得到三个嵌入向量表示，然后这三个表示加起来就是BERT的输入嵌入。
    这三个层都是通过学习得来的。
    '''

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = BertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    