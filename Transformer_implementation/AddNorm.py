import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(size)
        self.dropout =nn.Dropout(dropout)

    def forward(self, x, sublayer, **kwargs):
        out = x + sublayer(x, **kwargs)
        out = self.dropout(out)
        out = self.ln(x)
        return out
