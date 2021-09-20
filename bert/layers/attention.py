import torch.nn as nn


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        # 注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size / config.num_attention_heads == 0
        # 每个head的维度
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        ''''''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.query(hidden_states)
        mixed_value_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = 
