import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_size, ffn_size, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, emb_size),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.mlp(x)
