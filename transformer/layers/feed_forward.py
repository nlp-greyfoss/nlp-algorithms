import torch.nn as nn
import torch


class PositionWiseFeedForward(nn.Module):
    '''
    实现FFN网路
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # 将输入转换为d_ff维度
        self.w_1 = nn.Linear(d_model, d_ff)
        # 将d_ff转换回d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
