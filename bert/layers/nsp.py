import torch.nn as nn


class NextSentencePrediction(nn.Module):
    '''
    下一句预测：is_next或is_not_next
    '''

    def __init__(self, d_model):
        '''
        :param d_model: 隐藏层大小，即BERT模型的输出大小
        '''
        super().__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # ?x[:, 0]
        return self.softmax(self.linear(x[:, 0]))
