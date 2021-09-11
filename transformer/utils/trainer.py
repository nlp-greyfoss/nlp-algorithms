import time

import torch
import numpy as np

from torch.autograd import Variable

from transformer.utils.functions import subsequent_mask


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
        # 计算loss
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


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    # 保持增大批大小并计算token + 填充的总数
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


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


global max_src_in_batch, max_tgt_in_batch
