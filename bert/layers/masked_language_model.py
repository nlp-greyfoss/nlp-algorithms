import torch.nn as nn


class MaskedLanguageModel(nn.Module):
    '''
    从屏蔽的输入序列中预测原来的标记
    '''

    def __init__(self, d_model, vocab_size):
        '''
        :param d_model: 隐藏层大小，即BERT模型的输出大小
        :param vocab_size: 词表大小
        '''
        super.__init__()
        # 一个线性层
        self.linear = nn.Linear(d_model, vocab_size)
        # 加上softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x的大小？
        return self.softmax(self.linear(x))
