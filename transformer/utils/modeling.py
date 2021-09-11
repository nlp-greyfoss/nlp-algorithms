import copy

import torch.nn as nn

from transformer.layers.decoder import Decoder
from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.embeddings import Embeddings
from transformer.layers.encoder import Encoder
from transformer.layers.encoder_decoder import EncoderDecoder
from transformer.layers.encoder_layer import EncoderLayer
from transformer.layers.feed_forward import PositionWiseFeedForward
from transformer.layers.generator import Generator
from transformer.layers.multi_head_attention import MultiHeadedAttention
from transformer.layers.positional_encoding import PositionalEncoding


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    '''
    构建模型
    :param src_vocab: 源词典大小
    :tgt_vocab: 目标词典大小
    :param N: 叠加层数
    :param d_model: 模型的维度
    :param d_ff: FF网络中内层的维度
    :param h: 多头注意力的head数量
    :param dropout: dropout的比率
    '''
    c = copy.deepcopy  # 深克隆
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # 参数的初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model


