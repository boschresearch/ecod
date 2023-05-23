# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from collections import OrderedDict

import torch

from torchvision.ops import FeaturePyramidNetwork

from ecod.models.backbones.convnet import ConvNetworkWithPooling
from ecod.models.necks.conv import ConvNeck
from ecod.models.heads.ssd import SSDBoxHead
from ecod.data.box2d.priors import patch_prior_params_from_conv_net, check_prior_params


class Seq2ChannelsODHeads(torch.nn.Module):
    def __init__(self, args_dict, shape):
        super().__init__()
        assert len(shape) == 3, f"shape has to be (channels, height, width), but is {shape}"
        self.n_seqs = args_dict["seq_n_buffers"]
        self.inp_shape = shape
        self.use_fpn = args_dict["ssd_use_fpn"]
        self.n_predictions = args_dict["train_n_predictions"]
        self.n_timesteps = args_dict["shape_t"][0]
        self.neck =  ConvNeck(shape[0] * self.n_seqs, args_dict["hidden_dims"])
        self.neck_output_shapes = self.neck.get_output_shape([shape[0] * self.n_seqs, *shape[1:]])
        out_channels = [ss[0] for ss in self.neck_output_shapes]
        feature_maps = [ss[1] for ss in self.neck_output_shapes]
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(out_channels, out_channels[-1], None)
            self.fpn_names = [f"feat_{ii}" for ii in range(len(out_channels))]
            # overwrite out_channels because will use output of FPN
            out_channels = [out_channels[-1] for _ in range(len(out_channels))]
        args_dict = patch_prior_params_from_conv_net(args_dict, out_channels, feature_maps)
        check_prior_params(args_dict)
        self.ssd_box_head = SSDBoxHead(args_dict)

    def get_hidden_state(self, batch_size, device):
        ns = self.n_seqs
        hx = torch.zeros(
            (batch_size, ns - 1, *self.inp_shape),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        return hx

    def get_feature_list(self, inp):
        # (b, n_t, c, h, w) -> (b, n_t * c, h, w) -> [(b, c_i, h_i, w_i) for i in range(len(heads))]
        feature_list = self.neck(inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2], *inp.shape[3:]))
        if self.use_fpn:
            feature_dict = self.fpn(OrderedDict([(name, ff) for name, ff in zip(self.fpn_names, feature_list)]))
            feature_list = [feature_dict[name] for name in self.fpn_names]
        return feature_list

    def forward(self, inp, hx=None):
        if self.training:
            return self.forward_train(inp, hx)
        else:
            return self.forward_test(inp, hx)

    def forward_train(self, inp, hx):
        first_idx = self.n_timesteps - self.n_predictions - (self.n_seqs - 1)
        # as this module does have finite memory, it doesn't make sense to waste compute if the sequence is longer
        # than the memory, as we will take the last results only
        if first_idx >= 0:
            inp_cat = inp[:, first_idx:]
        else:
            if hx is None:
                hx = self.get_hidden_state(inp.shape[0], inp.device)
            if first_idx > -self.n_seqs + 1:
                hx = hx[:, first_idx:]
            inp_cat = torch.cat([hx, inp], dim=1)
        return self._forward(inp_cat, self.n_predictions)

    def forward_test(self, inp, hx):
        if hx is None:
            hx = self.get_hidden_state(inp.shape[0], inp.device)
        inp_cat = torch.cat([hx, inp], dim=1)
        return self._forward(inp_cat, inp.shape[1])

    def _forward(self, inp_cat, n_preds):
        # sliding window over input and internal state
        cls_logits = []
        bbox_reg = []
        for ii in range(n_preds):
            inp_now = inp_cat[:, ii : ii + self.n_seqs]
            # (b, n_t, c, h, w) -> (b, n_t * c, h, w) -> [(b, c_i, h_i, w_i) for i in range(len(heads))]
            feature_list = self.get_feature_list(inp_now)
            # (b, n_priors, 4) and (b, n_priors, n_classes), or [dict for _ in range(batches)] during val/test
            out_list = self.ssd_box_head(feature_list)
            cls_logits.append(out_list[0])
            bbox_reg.append(out_list[1])
        # hidden state in this model is just the last n_seqs-1 features
        if self.n_seqs <= 1:
            # negative indexing doesn't work in this case, but this also means there is no internal state to be kept
            hx = self.get_hidden_state(inp_cat.shape[0], inp_cat.device)
        else:
            hx = inp_cat[:, -self.n_seqs + 1 :]
        return (torch.stack(cls_logits, 1), torch.stack(bbox_reg, 1)), hx
