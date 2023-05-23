# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging

import torch


def extend_params(channels, kernel_sizes, paddings, strides):
    if kernel_sizes is None:
        kernel_sizes = [3 for _ in channels]
    if paddings is None:
        paddings = [1 for _ in channels]
    if strides is None:
        strides = [1 for _ in channels]
    return kernel_sizes, paddings, strides


def set_is_pretrained(named_modules, pretrained):
    for name, mm in named_modules:
        if len([mm for mm in mm.parameters()]) > 0:
            mm.is_pretrained = pretrained


class ConvNetwork(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        channels,
        kernel_sizes=None,
        paddings=None,
        strides=None,
        dropout_prob=0.2,
        return_idxs=None,
    ):
        """
        Be a bit careful with return_idxs: e.g., [0,1,2] would return after first conv, relu, dropout, and not
        after first second and third block
        """
        super().__init__()
        self.return_idxs = return_idxs
        kernel_sizes, paddings, strides = extend_params(channels, kernel_sizes, paddings, strides)
        assert len(kernel_sizes) == len(paddings) and len(paddings) == len(strides), (
            f"channels, kernel_sizes, paddings and strides all need to have the same length,"
            f" but are {channels}, {kernel_sizes}, {paddings}, {strides}"
        )
        self.layers = torch.nn.ModuleList()
        for ii, (kk, pp, ss) in enumerate(zip(kernel_sizes, paddings, strides)):
            if ii > 0:
                in_chans = channels[ii - 1]
            self.layers.append(torch.nn.Conv2d(in_chans, channels[ii], kk, padding=pp, stride=ss))
            self.layers.append(torch.nn.ReLU())
            if dropout_prob > 0.0:
                self.layers.append(torch.nn.Dropout(p=dropout_prob))
        if any([kk > len(self.layers) - 1 for kk in return_idxs]):
            raise ValueError(f"return_idxs are {return_idxs} but have to be in range({len(self.layers)})")

    def forward(self, inp):
        out = inp
        outs = [] if -1 not in self.return_idxs else [inp]
        for ii, layer in enumerate(self.layers):
            out = layer(out)
            if self.return_idxs is not None:
                if ii in self.return_idxs:
                    outs.append(out)
        if self.return_idxs is None:
            return out
        else:
            return outs


class ConvNetworkWithPooling(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        channels,
        pooling_idxs,
        kernel_sizes=None,
        paddings=None,
        strides=None,
        dropout_prob=0.2,
        return_idxs=None,
    ):
        super().__init__()
        self.return_idxs = return_idxs
        self.pooling_idxs = pooling_idxs
        kernel_sizes, paddings, strides = extend_params(channels, kernel_sizes, paddings, strides)
        assert len(kernel_sizes) == len(paddings) and len(paddings) == len(strides), (
            f"channels, kernel_sizes, paddings and strides all need to have the same length,"
            f" but are {channels}, {kernel_sizes}, {paddings}, {strides}"
        )
        self.layers = torch.nn.ModuleList()
        for ii, (kk, pp, ss) in enumerate(zip(kernel_sizes, paddings, strides)):
            if ii > 0:
                in_chans = channels[ii - 1]
            self.layers.append(torch.nn.Conv2d(in_chans, channels[ii], kk, padding=pp, stride=ss))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(p=dropout_prob))
            if ii in pooling_idxs:
                self.layers.append(torch.nn.MaxPool2d(2))
        if any([kk > len(self.layers) - 1 for kk in return_idxs]):
            raise ValueError(f"return_idxs are {return_idxs} but have to be in range({len(self.layers)})")
        self.log_layers()

    def log_layers(self):
        log_string = "input"
        if -1 in self.return_idxs:
            log_string += "|"
        log_string += "->"
        for ii, layer in enumerate(self.layers):
            log_string += f"{layer}"
            if ii in self.return_idxs:
                log_string += "|"
            if ii != len(self.layers) - 1:
                log_string += "->"
        logger = logging.getLogger("CONVPOOL")
        logger.setLevel("INFO")
        logger.info(f"Set up neck as {log_string}")

    @torch.no_grad()
    def get_output_shapes(self, in_shape_wo_batch):
        inp = torch.zeros((1, *in_shape_wo_batch), device="cpu")
        return [ss.shape[1:] for ss in self.to("cpu")(inp)]

    def forward(self, inp):
        out = inp
        outs = [] if -1 not in self.return_idxs else [inp]
        for ii, layer in enumerate(self.layers):
            out = layer(out)
            if self.return_idxs is not None:
                if ii in self.return_idxs:
                    outs.append(out)
        if self.return_idxs is None:
            return out
        else:
            return outs
