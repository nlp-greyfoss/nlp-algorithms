from torch.utils.data import Dataset

import tqdm
import torch
import random


class BertDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding='utf-8', corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, 'r', encoding=encoding) as f:
            # 计算总行数
            if self.corpus_lines is None:
                for _ in tqdm.tqdm(f, desc='Loading Dataset', total=corpus_lines):
                    self.corpus_lines += 1

            self.lines = [line[:-1].split('\t')
                          for line in tqdm.tqdm(f, desc='Loading Dataset', total=corpus_lines)]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        sent_A, sent_B, is_next_label = self.random_sent(item)
        random_A, A_label = self.random_word(sent_A)
        random_B, B_label = self.random_word(sent_B)

        # [CLS] token = SOS token, [SEP] token = EOS token
        sent_A = [self.vocab.sos_index] + random_A + [self.vocab.eos_index]
        sent_B = random_B + [self.vocab.eos_index]

        A_label = [self.vocab.pad_index] + A_label + [self.vocab.pad_index]
        B_label = B_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(sent_A))] + [2 for _ in range(len(sent_B))])[:self.seq_len]
        bert_input = (sent_A + sent_B)[:self.seq_len]
        bert_label = (A_label + B_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
            'is_next': is_next_label
        }

        return output

    def random_word(self, sentence):
        '''
        for mlm
        :param sentence:
        :return:
        '''
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                # 只随机一次，将prob除以0.15，又得到[0,1)之间的值，当成概率值
                prob /= 0.15

                # 80%的概率将token改成mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% 将token改成随机的token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% 保持不变
                else:
                    # 把token转换为id
                    tokens[i] = self.vocab.word2id.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.word2id.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.word2id.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        sent_A, sent_B = self.get_corpus_line(index)
        # output_text, label(notNext:0, isNext:1)
        if random.random() > 0.5:
            return sent_A, sent_B, 1
        else:
            return sent_A, self.get_random_line(), 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]
