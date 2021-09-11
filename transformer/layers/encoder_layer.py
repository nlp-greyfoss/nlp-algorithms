import torch.nn as nn

from transformer.layers.sublayer_connection import SublayerConnection
from transformer.utils.functions import clones


class EncoderLayer(nn.Module):
    '''
    编码器是由self-attention和FFN(全连接层神经网络)组成，其中self-attention和FNN分别用SublayerConnection封装
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        :param: size: 模型的大小
        :param: self_attn: 注意力层
        :param: feed_forward: 全连接层
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 编码器层共有两个子层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
