import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # 词典大小 嵌入大小
        self.d_model = d_model

    def forward(self, x):
        '''
        x: [batch_size, input_len]
        '''
        # 把得到的词嵌入向量乘以sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)  # [batch_size, input_len, d_model]
