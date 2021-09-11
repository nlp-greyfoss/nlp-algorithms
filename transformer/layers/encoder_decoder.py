import torch.nn as nn


class EncoderDecoder(nn.Module):
    '''
    一个标准的Encoder-Decoder架构
    '''

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        '''
        构建编码器-解码器结构
        :param encoder: 编码器实例
        :param decoder: 解码器实例
        :param src_embed: 源嵌入层实例
        :param tgt_embed: 目标嵌入层实例
        :param generator: 生成器实例
        '''
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
