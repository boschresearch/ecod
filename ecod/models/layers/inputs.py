# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch


class AvgPool2dChannels(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if hasattr(kernel_size, "__len__") else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = stride
        fill_value = 1.0 / (self.kernel_size[0] * self.kernel_size[1] * in_channels)
        self.register_buffer(
            "weight",
            torch.full((out_channels, in_channels, *self.kernel_size), fill_value),
        )

    def forward(self, inp):
        out = torch.nn.functional.conv2d(inp, self.weight, bias=None, stride=self.stride, padding=self.padding)
        return out


class ImageNetInputNormalizer(torch.nn.Module):
    """When using torchvision.models, need to normalize input (see
    https://pytorch.org/vision/stable/models.html )
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("means", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("stds", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def __call__(self, inp):
        return (inp - self.means) / self.stds
