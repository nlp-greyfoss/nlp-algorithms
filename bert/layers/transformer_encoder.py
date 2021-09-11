import torch.nn as nn

from transformer.layers.multi_head_attention import MultiHeadedAttention
from transformer.layers.feed_forward import PositionWiseFeedForward
from transformer.layers.sublayer_connection import SublayerConnection
from transformer.utils.functions import clones


class TransformerEncoder(nn.Module):
    '''
    BERT使用的TransformerEncoder，Transformer的编码器。
    多头注意力+前馈神经网络+子层连接+Dropout
    '''

    def __init__(self, d_model, d_ff, attn_heads, dropout=0.1):
        '''
        :param d_model: 隐藏层大小，即BERT模型的输出大小
        :param d_ff: 前馈神经网络的隐藏层大小，通常为 4 * d_model
        :param attn_heads: 多头注意力的头数
        :param dropout: dropout比率，通常为0.1
        '''

        super.__init__()
        self.attention = MultiHeadedAttention(attn_heads, d_model, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.sublayer[1](x, self.feed_forward)
        return self.dropout(x)
