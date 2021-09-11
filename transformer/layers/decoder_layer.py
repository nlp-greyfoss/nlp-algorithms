import torch.nn as nn

from transformer.layers.sublayer_connection import SublayerConnection
from transformer.utils.functions import clones


class DecoderLayer(nn.Module):
    '''
    解码器由self-attn(第一个子层),src-attn(连接Encoder的子层)和全连接层(第三个子层)组成
    '''

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        :param: x tgt的嵌入 [batch_size, tgt_len, emb_size]
        :param: memory 编码器的输出 [batch_size, input_len, emb_size]
        :param: src_mask 源数据mask [batch_size, 1, input_len]
        :param: tgt_mask 目标数据mask [batch_size, tgt_len, tgt_len]
        '''
        m = memory
        # 第一个注意力层，query,key,value来自同一个输入。tgt_mask用于屏蔽后面的位置
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个注意里层，key,value来自编码器的输出，query来自上一层的输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
