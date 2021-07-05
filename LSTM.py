import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import math

from torch.nn import init


class simpleLSTM(nn.Module):

    #基于RNNBase源码进行初始化
    def __init__(self, input_size, hidden_size,num_layers=1, bias=True, batch_first=False,dropout=0., bidirectional=False,SEQ_LENGTH = 30):
        super(simpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.SEQ_LENGTH = SEQ_LENGTH
        # 单向LSTM
        # self.bidirectional = bidirectional
        # num_directions = 2 if bidirectional else 1
        num_directions = 1

        #LSTM 输入门，输出门，记忆细胞，遗忘门
        gate_size = 4 * hidden_size

        self._all_weights = []

        #层形状初始化
        for layer in range(num_layers):
            for direction in range(num_directions):
                #层输入形状运算
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                #可训练参数
                # w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                # w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                # b_ih = Parameter(torch.Tensor(gate_size))
                # # Second bias vector included for CuDNN compatibility. Only one
                # # bias vector is needed in standard definition.
                # b_hh = Parameter(torch.Tensor(gate_size))

                #输入门
                self.w_ii = Parameter(torch.Tensor(hidden_size,layer_input_size))
                self.w_hi = Parameter(torch.Tensor(hidden_size,hidden_size))
                self.b_ii = Parameter(torch.Tensor(hidden_size,1))
                self.b_hi = Parameter(torch.Tensor(hidden_size,1))

                #遗忘门
                self.w_if = Parameter(torch.Tensor(hidden_size, layer_input_size))
                self.w_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
                self.b_if = Parameter(torch.Tensor(hidden_size, 1))
                self.b_hf = Parameter(torch.Tensor(hidden_size, 1))

                #输出门
                self.w_io = Parameter(torch.Tensor(hidden_size, layer_input_size))
                self.w_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
                self. b_io = Parameter(torch.Tensor(hidden_size, 1))
                self.b_ho = Parameter(torch.Tensor(hidden_size, 1))

                #记忆细胞
                self.w_ig =  Parameter(torch.Tensor(hidden_size, layer_input_size))
                self.w_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
                self.b_ig = Parameter(torch.Tensor(hidden_size, 1))
                self.b_hg = Parameter(torch.Tensor(hidden_size, 1))

                self.reset_parameters()

                # layer_params = (w_ii, w_hi, b_ii, b_hi,w_if, w_hf, b_if, b_hf,w_io, w_ho, b_io, b_ho,w_ig, w_hg, b_ig, b_hg)

                # suffix = '_reverse' if direction == 1 else ''
                # param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                # if bias:
                #     param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                # param_names = [x.format(layer, suffix) for x in param_names]
                #
                # for name, param in zip(param_names, layer_params):
                #     setattr(self, name, param)
                # self._all_weights.append(param_names)

    #RNNBase 的参数初始化方法
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    #前向运算
    def forward(self, inputs, state):

        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        seq_size = self.SEQ_LENGTH
        for t in range(seq_size):
            x = inputs[:, t, :].t()

            # input gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +
                              self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # print(hidden_seq.size())
        return hidden_seq, (h_next_t, c_next_t)

