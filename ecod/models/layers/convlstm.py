"""ConvLSTM implementation.

Adapted from https://github.com/aserdega/convlstmgru
(MIT License)
"""
__copyright__ = """

MIT License

Copyright (c) 2020 Andriy Serdega

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.nn as nn

import math


class ConvLSTMCell(nn.Module):

    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 bias=True,
                 activation=torch.tanh,
                 peephole=False,
                 batchnorm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.activation = activation
        self.peephole = peephole
        self.batchnorm = batchnorm

        if peephole:
            self.Wci = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            self.Wcf = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))
            self.Wco = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.reset_parameters()

    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state

        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.peephole:
            i = torch.sigmoid(cc_i + self.Wci * c_prev)
            f = torch.sigmoid(cc_f + self.Wcf * c_prev)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)

        g = self.activation(cc_g)
        c_cur = f * c_prev + i * g

        if self.peephole:
            o = torch.sigmoid(cc_o + self.Wco * c_cur)
        else:
            o = torch.sigmoid(cc_o)

        h_cur = o * self.activation(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, device='cpu'):
        state = (
            torch.zeros((batch_size, self.hidden_dim, self.height, self.width), device=device, dtype=torch.float32),
            torch.zeros((batch_size, self.hidden_dim, self.height, self.width), device=device, dtype=torch.float32),
        )
        return state

    def reset_parameters(self):
        #self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('tanh'))
        self.conv.bias.data.zero_()

        if self.batchnorm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()
        if self.peephole:
            std = 1. / math.sqrt(self.hidden_dim)
            self.Wci.data.uniform_(0, 1)  #(std=std)
            self.Wcf.data.uniform_(0, 1)  #(std=std)
            self.Wco.data.uniform_(0, 1)  #(std=std)


class ConvLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 batch_first=False,
                 bias=True,
                 activation=torch.tanh,
                 peephole=False,
                 batchnorm=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation = self._extend_for_multilayer(activation, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(activation) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.batch_dim = 0 if self.batch_first else 1
        self.time_dim = 1 if self.batch_first else 0

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(input_size=(self.height, self.width),
                             input_dim=cur_input_dim,
                             hidden_dim=self.hidden_dim[i],
                             kernel_size=self.kernel_size[i],
                             bias=self.bias,
                             activation=activation[i],
                             peephole=peephole,
                             batchnorm=batchnorm))

        self.cell_list = nn.ModuleList(cell_list)

        self.reset_parameters()

    def forward(self, input, hidden_states=None):
        """

        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            list of 2-tuple of tensors (b, c_hidden, h, w)
        Returns
        -------
        last_state_list, layer_output
        """
        if hidden_states is None:
            bs = input.shape[self.batch_dim]
            hidden_states = self.get_init_states(bs, input.device)

        cur_layer_input = torch.unbind(input, dim=self.time_dim)
        seq_len = len(cur_layer_input)
        # all_hidden_states[l][t] is the state at layer l and time t
        all_hidden_states = [[] for _ in self.hidden_dim]
        last_cell_states = []
        for layer_idx in range(self.num_layers):
            hidden_state = hidden_states[layer_idx]
            for t in range(seq_len):
                hidden_state = self.cell_list[layer_idx](input=cur_layer_input[t], prev_state=hidden_state)
                all_hidden_states[layer_idx].append(hidden_state[0])
            last_cell_states.append(hidden_state[1])
            cur_layer_input = all_hidden_states[layer_idx]

        all_hidden_states = [torch.stack(hh, dim=self.time_dim) for hh in all_hidden_states]
        if self.batch_first:
            current_states = [(hh[:, -1], cc) for hh, cc in zip(all_hidden_states, last_cell_states)]
        else:
            current_states = [(hh[-1], cc) for hh, cc in zip(all_hidden_states, last_cell_states)]
        return all_hidden_states, current_states

    def reset_parameters(self):
        for c in self.cell_list:
            c.reset_parameters()

    def get_init_states(self, batch_size, device='cpu'):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`Kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def get_output_shapes(self, input_shape, device='cpu'):
        with torch.no_grad():
            input_shape = [1] + list(input_shape)
            _, last_state_list = self.to(device)(torch.zeros(input_shape, device=device))
            output_shapes = [out[0].shape[1:] for out in last_state_list]
        return output_shapes
