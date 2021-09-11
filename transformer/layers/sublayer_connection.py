import torch.nn as nn

from transformer.layers.layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    '''
    残差连接然后接层归一化
    为了简化代码，先进行层归一化
    '''

    def __init__(self, size, dropout):
        '''
        :param size: 模型的维度，原文中统一为512
        :param dropout: Dropout的比率，原文中为0.1
        '''
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        :param x:子层的输入
        :param sublayer: 子层
        应用残差连接到任何同样大小的子层
        '''
        return x + self.dropout(sublayer(self.norm(x)))
