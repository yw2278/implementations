import torch.nn as nn
from Attention import MultiHeadAttention
from AddNorm import AddNorm
from FeedForward import FeedForward
import numpy as np


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.block_size = config.seq_length
        self.position_embedding = nn.Parameter(np.zeros(1, config.seq_length, config.d_size))
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_size, padding_idx=config.pad)
        self.attn = MultiHeadAttention(config.nhead, config.d_size, config.k_size, config.v_size, config.dropout)
        self.add_norm = AddNorm(config.d_size, config.dropout)
        self.ffn = FeedForward(config.d_size, 4*config.d_size, config.dropout)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x, mask):
        b, t = x.size()
        assert t < self.block_size
        out = self.position_embedding[:, :t, :] + self.token_embedding(x)
        out = self.add_norm(out, self.attn, y=out, mask=mask)
        out = self.add_norm(out, self.ffn)

        return out

