import torch.nn as nn
import torch
import math

class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size):
        # input_size: B, T, C
        super(LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.W1 = nn.Parameter(torch.Tensor(input_size, embedding_size*4))
        self.W2 = nn.Parameter(torch.Tensor(embedding_size, embedding_size*4))
        self.b1 = nn.Parameter(torch.Tensor(embedding_size*4))
        self.b2 = nn.Parameter(torch.Tensor(embedding_size*4))
        self.init_weights()

    def init_weights(self):
        stdv = 1/math.sqrt(self.embedding_size)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, initial_state):
        B, T, C = x.size()
        es = self.embedding_size
        if not initial_state:
            h, c = torch.zeros(B, es).to(x.device), torch.zeros(B, es).to(x.device)
        else:
            h, c = initial_state
        out_h = []
        for t in range(T):
            xt = x[:, t, :]
            v = xt @ self.W1 + self.b1 + h @ self.W2 + self.b2
            f = torch.sigmoid(v[:, :es])
            i = torch.sigmoid(v[:, es:es*2])
            o = torch.sigmoid(v[:, es*2:es*3])
            c_hat = torch.tanh(v[:, es*3:])

            c = f*c+i*c_hat
            h = o*torch.tanh(c)

            out_h.append(h)
        out_h_1 = torch.cat(out_h, dim=-1)
        out_h_2 = torch.stack(out_h)
        return out_h_1, out_h_2, (h, c) # T, B, es


if __name__ == '__main__':
    b, t, i, hs = 2,3,4,5
    inp = torch.randn(b, t, i)
    c0 = torch.randn(b, hs)
    h0 = torch.randn(b, hs)

    lstm = nn.LSTM(i, hs, batch_first=True)
    output, (hf, cf) = lstm(inp, (h0.unsqueeze(0), c0.unsqueeze(0)))

    my_lstm = LSTM(i, hs)
    out, out2,(out_h, out_c) = my_lstm(inp, (h0, c0))


    print('my:', out.shape)
    print('my2', out2.shape)






