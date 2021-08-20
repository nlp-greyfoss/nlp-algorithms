import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn  # # seaborn在可视化self-attention的时候用的到

seaborn.set_context(context="talk")


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


def clones(module, N):
    '''
    生成N个相同的层
    '''
    # ModuleList和Python普通列表一样索引，但是里面的模型会被合理的注册到网络中
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 编码器

class Encoder(nn.Module):
    '''
    Encoder堆叠了N个相同的层，下层的输出当成上层的输入
    '''

    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        '''
        依次将输入和mask传递到每层，why encoder needs mask
        :param x: [batch_size, input_len, emb_size]
        '''
        for layer in self.layers:
            # 下层的输出当成上层的输入
            x = layer(x, mask)
        # 最后进行层归一化
        return self.norm(x)


class LayerNorm(nn.Module):
    '''
    构建一个层归一化模块
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        :param x: [batch_size, input_len, emb_size]
        '''
        # 计算最后一个维度的均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    '''
    残差连接然后接层归一化
    为了简化代码，先进行层归一化
    '''

    def __init__(self, size, dropout):
        '''
        :param size: 模型的维度，原文中统一为512
        :param dropout: Dropout的比率，原文中为0.1
        '''
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        应用残差连接到任何同样大小的子层
        '''
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    '''
    编码器是由self-attention和FFN(全连接层神经网络)组成，其中self-attention和FNN分别用SublayerConnection封装
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        :param: size: 模型的大小
        :param: self_attn: 注意力层
        :param: feed_forward: 全连接层
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 编码器层共有两个子层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 解码器

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        :param: x tgt的嵌入 [batch_size, tgt_len, emb_size]
        :param: memory 编码器的输出 [batch_size, input_len, emb_size]
        :param: src_mask 源数据mask [batch_size, 1, input_len]
        :param: tgt_mask 目标数据mask [batch_size, tgt_len, tgt_len]
        '''
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    '''
    解码器由self-attn(第一个子层),src-attn(连接Encoder的子层)和全连接层(第三个子层)组成
    '''

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        :param: x tgt的嵌入 [batch_size, tgt_len, emb_size]
        :param: memory 编码器的输出 [batch_size, input_len, emb_size]
        :param: src_mask 源数据mask [batch_size, 1, input_len]
        :param: tgt_mask 目标数据mask [batch_size, tgt_len, tgt_len]
        '''
        m = memory
        # 第一个注意力层，query,key,value来自同一个输入。tgt_mask用于屏蔽后面的位置
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个注意里层，key,value来自编码器的输出，query来自上一层的输出
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    '''
    为后续输出序列的位置添加屏蔽
    :param size: 输出序列长度
    '''
    attn_shape = (1, size, size)
    # 主对角线上移一位，主对角线下的元素全为0
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    '''
    计算缩放点乘注意力
    :param query:  [batch_size, self.h, input_len, self.d_k]
    :param key:    [batch_size, self.h, input_len, self.d_k]
    :param value:  [batch_size, self.h, input_len, self.d_k]
    '''
    d_k = query.size(-1)
    # query: [batch_size, self.h, input_len, self.d_k]
    # key.transpose: [batch_size, self.h, self.d_k, input_len]
    # 此时就是批矩阵相乘 固定batch_size, self.h  -> input_len, self.d_k  x self.d_k, input_len = input_len, input_len
    # -> [batch_size, self.h, input_len, input_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 源序列也需要mask，因为批次内语句长短不一，对于短的语句，就需要填充<pad>字符
    if mask is not None:
        # 根据mask句子，把屏蔽的位置填-1e9，然后计算softmax的时候，-1e9的位置就被计算为0
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)  # 经过softmax得到注意力权重
    if dropout:
        p_attn = dropout(p_attn)
    #  [batch_size, self.h, input_len, input_len]  x  [batch_size, self.h, input_len, self.d_k]
    # -> [batch_size, self.h, input_len, self.d_k]
    return torch.matmul(p_attn, value), p_attn  # 返回最终的输出 和 注意力权重(可用于绘图)


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 注册为一个缓存，但是不是模型的参数，默认随模型其他参数一起保存

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


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


class Batch:
    """
    保持带有Mask的批次数据对象
    """

    def __init__(self, src, trg=None, pad=0):
        '''
         :param src: 源数据 [batch_size, input_len]
         :param trg: 目标数据 [batch_size, input_len]
        '''
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # 在倒数第二个位置加入一个维度
        if trg is not None:
            # 机器翻译任务在输出时，需要根据左边所有的序列，去预测下一个序列。
            self.trg = trg[:, :-1]  # 去掉最后一个，可用于teacher forcing，此时self.trg维度变成了[batch_size, input_len-1]
            self.trg_y = trg[:, 1:]  # 去掉第一个，构建目标输出

            # # 目标mask的目的是防止当前位置注意到后面的位置 [batch_size,input_len-1, input_len-1]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()  # 实际单词(不包括填充词)数量

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 在倒数第二个位置加入一个维度
        # type_as 把调用tensor的类型变成给定tensor的
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        )
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    '''
    标准的训练和打印函数
    '''
    start = time.time()
    total_tokens = 0  # 总单词数
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print(f"Epoch Step: {i} Loss: {loss / batch.ntokens} Tokens per Sec: {tokens / elapsed}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


class NoamOpt:
    '''封装了学习率的优化器'''

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 更新学习率
        self._rate = rate
        self.optimizer.step()  # 更新其他参数

    def rate(self, step=None):
        '''实现上面公式中的学习率'''
        if not step:
            step = self._step

        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, torch.tensor(target.data.unsqueeze(1), dtype=torch.int64), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    '''
    :param batch: 批次大小
    :param nbatches: 生成批次次数
    生成随机数据，其中src和tgt是一样的
    '''
    for i in range(nbatches):
        # 从[1,V]之间生成随机数，生成的大小为 batch x 10, 每个批次有10个随机数
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data * norm


if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))
