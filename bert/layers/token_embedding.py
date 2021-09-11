import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, d_vocab, d_embed=512):
        super().__init__(d_vocab, d_embed, padding_idx=0)