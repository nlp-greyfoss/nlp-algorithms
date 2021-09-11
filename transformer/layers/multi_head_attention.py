import torch.nn as nn

from transformer.utils.functions import clones, attention


class MultiHeadedAttention(nn.Module):
    '''
    多头注意力机制实现
    '''

    def __init__(self, h, d_model, dropout=0.1):
        '''
        输入维度和head数量h
        '''
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        # d_k是每个head的维度
        self.d_k = d_model // h
        self.h = h
        # 四个线性层，三个在输入端，一个在输出端
        # 在计算注意力之前先将query,key,value进行线性变换效果更好
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 同样的mask应用到所有h个head
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)  # 批次大小

        # 1) 在批次内对query,key,value进行线性运算，分别转换成h个d_k维度的query,key,value：维度 d_model => h x d_k,
        # 对self.linears与(query,key,value)进行zip，相当于分别把query,key,value喂给前三个线性层，得到线性变换后的query,key,value
        # 如 query: [batch_size, input_len, d_model] -> 线性层 ->  [batch_size, input_len, d_model]
        # -> view -> [batch_size, input_len, self.h, self.d_k] -> transpose -> [batch_size, self.h, input_len, self.d_k]
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) 对批次内所有线性变换后的向量调用注意力函数
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) 通过view执行类似连接Z的操作，然后应用到最后一个线性层
        # view方法需要tensor是连续的，因此调用contiguous方法
        # x : [batch_size, self.h, input_len, self.d_k] -> transpose -> [batch_size, input_len, self.h, self.d_k]
        # -> view -> [batch_size, input_len, self.h*self.d_k]
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x)
