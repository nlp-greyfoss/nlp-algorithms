import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    '''
    构建一个层归一化模块
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        :param x: [batch_size, input_len, emb_size]
        '''
        # 计算最后一个维度的均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
