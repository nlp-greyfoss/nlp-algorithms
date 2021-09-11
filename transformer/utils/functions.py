import torch.nn as nn
import torch
import numpy as np
import copy
import math

def clones(module, N):
    '''
    生成N个相同的层
    '''
    # ModuleList和Python普通列表一样索引，但是里面的模型会被合理的注册到网络中
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    '''
    为后续输出序列的位置添加屏蔽
    :param size: 输出序列长度
    '''
    attn_shape = (1, size, size)
    # 主对角线上移一位，主对角线下的元素全为0
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    '''
    计算缩放点乘注意力
    :param query:  [batch_size, self.h, input_len, self.d_k]
    :param key:    [batch_size, self.h, input_len, self.d_k]
    :param value:  [batch_size, self.h, input_len, self.d_k]
    '''
    d_k = query.size(-1)
    # query: [batch_size, self.h, input_len, self.d_k]
    # key.transpose: [batch_size, self.h, self.d_k, input_len]
    # 此时就是批矩阵相乘 固定batch_size, self.h  -> input_len, self.d_k  x self.d_k, input_len = input_len, input_len
    # -> [batch_size, self.h, input_len, input_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 源序列也需要mask，因为批次内语句长短不一，对于短的语句，就需要填充<pad>字符
    if mask is not None:
        # 根据mask句子，把屏蔽的位置填-1e9，然后计算softmax的时候，-1e9的位置就被计算为0
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)  # 经过softmax得到注意力权重
    if dropout:
        p_attn = dropout(p_attn)
    #  [batch_size, self.h, input_len, input_len]  x  [batch_size, self.h, input_len, self.d_k]
    # -> [batch_size, self.h, input_len, self.d_k]
    return torch.matmul(p_attn, value), p_attn  # 返回最终的输出 和 注意力权重(可用于绘图)
