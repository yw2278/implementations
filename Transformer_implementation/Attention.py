import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, d_size, k_size, v_size, dropout):
        super().__init__()
        self.nhead = nhead
        self.k_size = k_size
        self.v_size = v_size
        self.query = nn.Linear(d_size, k_size, bias=False)
        self.key = nn.Linear(d_size, k_size, bias=False)
        self.value = nn.Linear(d_size, v_size, bias=False)

        self.proj = nn.Linear(v_size, d_size, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, y, mask):
        # B,T,C -> B,T,N,C/N
        B, T, C = x.size()
        # y: B, T, C
        assert  self.k_size % self.nhead  == 0 and self.v_size % self.nhead  == 0

        k = self.key(x).view(B, T, self.nhead, self.k_size/self.nhead).transpose(2,1) # B, N, T, C/N
        q = self.query(x).view(B, T, self.nhead, self.k_size/self.nhead).transpose(2,1)
        v = self.value(y).view(B, T, self.nhead, self.v_size/self.nhead).transpose(2,1)

        alpha = (q @ k.transpose(3,2))/np.sqrt(self.k_size/self.nhead) # B, N, T, T
        if mask: #B, 1, T, T
            mask = mask.unsqueeze(1)
            alpha.masked_fill(mask == 0, float('-inf'))
        attn = nn.Softmax(alpha, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v # B, N, T, C/N
        out = out.transpose(2, 1).contiguous().view(B, T, self.v_size)
        out = self.proj_drop(self.proj(out))

        return out

