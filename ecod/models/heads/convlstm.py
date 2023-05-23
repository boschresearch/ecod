# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

from ecod.models.layers.convlstm import ConvLSTM

from ecod.models.heads.ssd import SSDBoxHead
from ecod.data.box2d.priors import patch_prior_params_from_conv_net, check_prior_params


class ConvLSTMNeck(torch.nn.Module):
    def __init__(self, args_dict, shape):
        super().__init__()
        assert len(shape) == 3, f"shape has to be (channels, height, width), but is {shape}"
        hidden_dims = args_dict["hidden_dims"]
        #self.n_buffers = args_dict["seq_n_buffers"]
        self.n_timesteps = args_dict["n_timesteps"]
        self.shape = shape
        self.conv_lstm = ConvLSTM(
            shape[1:],
            shape[0],
            hidden_dims,
            kernel_size=[(3, 3)] * len(hidden_dims),
            num_layers=len(hidden_dims),
            batch_first=True,
            bias=True,
            activation=[torch.tanh] * len(hidden_dims),
            peephole=False,
            batchnorm=False,
        )
        # FIXME: maybe input shape is wrong, check
        self.conv_out_shapes = self.get_output_shape([args_dict["seq_n_buffers"], *shape])

    def get_output_shape(self, input_shape):
        lstm_shape = self.conv_lstm.get_output_shapes([self.n_timesteps, *self.shape])
        # return [input_shape] + lstm_shape
        return lstm_shape

    def get_hidden_state(self, batch_size, device):
        return self.conv_lstm.get_init_states(batch_size, device=device)

    def forward(self, inp, hx):
        if hx is None:
            hx = self.get_hidden_state(inp.shape[0], inp.device)
        all_hidden_states, current_states = self.conv_lstm(inp, hx)
        return all_hidden_states, current_states


class ConvLSTMODHeads(torch.nn.Module):
    def __init__(self, args_dict, shape):
        super().__init__()
        self.neck = ConvLSTMNeck(args_dict, shape)
        conv_out_shapes = self.neck.conv_out_shapes
        out_channels = [ss[0] for ss in conv_out_shapes]
        feature_maps = [ss[1] for ss in conv_out_shapes]
        args_dict = patch_prior_params_from_conv_net(args_dict, out_channels, feature_maps)
        check_prior_params(args_dict)
        self.ssd_box_head = SSDBoxHead(args_dict)

    def get_hidden_state(self, batch_size, device):
        return self.neck.conv_lstm.get_init_states(batch_size, device=device)

    def forward(self, inp, hx):
        if hx is None:
            hx = self.get_hidden_state(inp.shape[0], inp.device)
        cls_logits = []
        bbox_reg = []
        all_hidden_states, hx = self.neck(inp, hx)
        for time in range(inp.shape[1]):
            features_list = [hh[:, time] for hh in all_hidden_states]
            out_list = self.ssd_box_head(features_list)
            cls_logits.append(out_list[0])
            bbox_reg.append(out_list[1])
        return (torch.stack(cls_logits, 1), torch.stack(bbox_reg, 1)), hx
