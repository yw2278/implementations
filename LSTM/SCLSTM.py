import torch
import torch.nn as nn

class SCLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, dropout, num_layers, k_size, output_size):
        super(SCLSTM, self).__init__()
        self.input_size = input_size
        self.d_size = embedding_size
        self.k_size = k_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.out_size = output_size

        self.wx = nn.ModuleList([nn.Linear(input_size + i*self.d_size, 4*self.d_size, bias=True) for i in range(num_layers)])
        self.wh = nn.ModuleList([nn.Linear(self.d_size, 4*self.d_size, bias=True) for _ in range(num_layers)])
        self.wr = nn.ModuleList([nn.Linear(input_size + i*self.d_size, self.k_size, bias=False) for i in range(num_layers)])
        self.wl = nn.ModuleList([nn.Linear(self.d_size, self.k_size, bias=False) for _ in range(num_layers)])
        self.wd = nn.Linear(self.k_size, self.d_size, bias=False)
        self.alpha = 1/num_layers

        self.proj = nn.Linear(self.d_size*num_layers, output_size)

    def run_layers(self, x, last_ht, last_dt, last_ct, training=False):
        # B, C
        xt = x
        ht_list, ct_list, dt_list = [], [], []
        for layer_id in range(self.num_layers):
            gate = self.wx[layer_id](xt) + self.wh[layer_id](last_ht[layer_id])
            f, i, o, c_ = (
                torch.sigmoid(gate[:, :self.d_size]),
                torch.sigmoid(gate[:, self.d_size:2*self.d_size]),
                torch.sigmoid(gate[:, 2*self.d_size:3*self.d_size]),
                torch.tanh(gate[:, 3*self.d_size:4*self.d_size])
            )
            gate_r = self.wr[layer_id](xt)
            for i in range(layer_id):
                gate_r += self.alpha*self.wl[i](last_ht[i])
            rt = torch.sigmoid(gate_r)
            dt = rt*last_dt[layer_id]
            ct = f*last_ct[layer_id] + i*c_ + torch.tanh(self.wd(dt))
            ht = last_ht[layer_id]*torch.tanh(ct)
            if training:
                ht = torch.dropout(ht, p=self.dropout, train=True)
            xt = torch.cat([xt, ht], dim=1)
            ht_list.append(ht)
            ct_list.append(ct)
            dt_list.append(dt)


        return ht_list, ct_list, dt_list

    def forward(self, inputs, init_keys, training=False):
        B, T, C = inputs.size()
        max_len = 55 if not training else T
        ht, ct, dt = [], [], []
        init_hidden = torch.Tensor(torch.zeros(B, self.d_size)).to(inputs.device)
        output = []
        for l in range(self.num_layers):
            ht.append(init_hidden)
            ct.append(init_hidden)
            dt.append(init_keys)
        for t in range(max_len):
            ht, ct, dt = self.run_layers(inputs, ht, ct, dt, training)
            out = self.proj(torch.cat(ht, dim=0))
            output.append(out)
        return torch.stack(output, dim=0).view(B, max_len, self.out_size)



if __name__ == '__main__':
    import numpy as np
    a = torch.randn(2,3)
    b = torch.randn(2,3)
    c = torch.stack([a,b], dim=0)
    print(c.shape)