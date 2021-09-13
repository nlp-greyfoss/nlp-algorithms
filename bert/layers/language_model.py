import torch.nn as nn

from bert.layers.masked_language_model import MaskedLanguageModel
from bert.layers.nsp import NextSentencePrediction


class BertLM(nn.Module):
    def __init__(self, bert, vocab_size):
        self.bert = bert
        self.nsp = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, token_type_ids=None):
        x = self.bert(x, token_type_ids)
        return self.nsp(x), self.mask_lm(x)
