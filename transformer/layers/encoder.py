import torch.nn as nn


# 编码器
from transformer.layers.layer_norm import LayerNorm
from transformer.utils.functions import clones


class Encoder(nn.Module):
    '''
    Encoder堆叠了N个相同的层，下层的输出当成上层的输入
    '''

    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        '''
        依次将输入和mask传递到每层，why encoder needs mask
        :param x: [batch_size, input_len, emb_size]
        '''
        for layer in self.layers:
            # 下层的输出当成上层的输入
            x = layer(x, mask)
        # 最后进行层归一化
        return self.norm(x)
