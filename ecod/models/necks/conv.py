# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch


class ConvNeckBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True,
    ):
        """ """
        super().__init__()
        layers = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
        ]
        if pool:
            layers += [torch.nn.MaxPool2d(2)]
            #layers += [torch.nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)]
        layers += [torch.nn.ReLU(), torch.nn.BatchNorm2d(out_channels)]
        self.model = torch.nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp)


class ConvNeck(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dims,
    ):
        """ """
        super().__init__()
        channels = [in_channels, *hidden_dims]
        self.blocks = torch.nn.ModuleList([])
        for idx, (in_channels, out_channels) in enumerate(zip(channels, channels[1:])):
            pool = idx > 0
            self.blocks.append(ConvNeckBlock(in_channels, out_channels, pool=pool))

    @torch.no_grad()
    def get_output_shape(self, in_shape_wo_batch):
        inp = torch.zeros((1, *in_shape_wo_batch), device="cpu")
        return [ss.shape[1:] for ss in self.to("cpu")(inp)]

    def forward(self, inp):
        outs = []
        out = inp
        for block in self.blocks:
            out = block(out)
            outs.append(out)
        return outs
