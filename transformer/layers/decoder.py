import torch.nn as nn

from transformer.layers.layer_norm import LayerNorm
from transformer.utils.functions import clones


# 解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        :param: x tgt的嵌入 [batch_size, tgt_len, emb_size]
        :param: memory 编码器的输出 [batch_size, input_len, emb_size]
        :param: src_mask 源数据mask [batch_size, 1, input_len]
        :param: tgt_mask 目标数据mask [batch_size, tgt_len, tgt_len]
        '''
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
