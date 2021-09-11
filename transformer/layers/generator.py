import torch.nn as nn
import torch


class Generator(nn.Module):
    '''定义标准的线性+softmax生成步骤'''

    def __init__(self, d_model, vocab):
        '''
        :param d_model: 模型大小
        :param vocab: 词典大小
        '''
        super(Generator, self).__init__()
        # 将d_model维度的向量线性变换到词典大小vocab维
        # 以输出每个单词的概率
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
