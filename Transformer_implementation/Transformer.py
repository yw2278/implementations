import torch.nn as nn
from Decoder import Decoder
from Encoder import Encoder
import torch
import torch.nn.functional as F

class Transformer(nn.Module):
    def __int__(self, config):
        self.decoder_part = nn.Sequential(*[Decoder(config) for _ in range(config.nlayers)])
        self.encoder_part = nn.Sequential(*[Encoder(config) for _ in range(config.nlayers)])
        self.register_buffer('decoder_mask', torch.tril(torch.zeros(1, 1, config.seq_length, config.seq_length)))
        self.proj = nn.Linear(config.d_size, config.vocab_size)

    def forward(self, x_in, x_out, mask):
        out = self.encoder_part(x_in, mask)
        out = self.decoder_part(x_out, out, self.decoder_mask)

        out = self.proj(out)
        out = F.softmax(out, dim=-1)

        return out




