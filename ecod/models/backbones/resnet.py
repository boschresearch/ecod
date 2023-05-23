# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import Bottleneck, BasicBlock


def _set_is_pretrained(named_modules, pretrained):
    for name, mm in named_modules:
        if len([mm for mm in mm.parameters()]) > 0:
            mm.is_pretrained = pretrained


class DenseNetBackbone(nn.Module):
    def __init__(self, name, pool=False, pretrained=True):
        super().__init__()
        self.pool = pool
        self.input_channels = 3
        self.pretrained = pretrained
        if name == "densenet201":
            self.model = torchvision.models.densenet201(pretrained=pretrained, progress=True)
            del self.model.classifier
        else:
            raise ValueError(f"name has to be in 'densenet201' but is {name}")
        self.set_is_pretrained()

    def set_is_pretrained(self):
        _set_is_pretrained(self.named_modules(), self.pretrained)

    def forward(self, inp):
        out = F.relu(self.model.features(inp))
        if self.pool:
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        return out


class ResNetBackbone(nn.Module):
    """extract backbone from torchvision.models"""

    def __init__(self, name, in_channels=3, pool=False, load_from=None):
        super().__init__()
        self.pool = pool
        self.input_channels = in_channels
        if name == "resnext50_32x4d":
            self.model = torchvision.models.resnext50_32x4d(weights=None, progress=True)
        elif name == "resnet18":
            self.model = torchvision.models.resnet18(weights=None, progress=True)
        elif name == "resnet50":
            self.model = torchvision.models.resnet50(weights=None, progress=True)
        elif name == "resnet101":
            self.model = torchvision.models.resnet101(weights=None, progress=True)
        else:
            raise ValueError(f"name has to be in 'resnet18', 'resnet50', 'resnet101', 'resnext50_32x4d' but is {name}")
        del self.model.fc
        self.patch_first_layer()
        _set_is_pretrained(self.named_modules(), load_from is not None)

    def patch_first_layer(self):
        if self.input_channels != 3:
            kernel_size = self.model.conv1.kernel_size
            stride = self.model.conv1.stride
            padding = self.model.conv1.padding
            out_channels = self.model.conv1.out_channels
            bias = self.model.conv1.bias
            groups = self.model.conv1.groups
            dilation = self.model.conv1.dilation
            self.model.conv1 = nn.Conv2d(
                self.input_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias,
                groups=groups,
                dilation=dilation,
            )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        return x


class ResNetBackboneCustom(nn.Module):
    """Small ResNet-like architecture with 32 channels after first conv"""

    def __init__(self, in_channels=3, layers=[3, 2, 1, 1], channels=[32, 64, 128, 256]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 32
        self.groups = 1
        self.base_width = 64
        self.set_is_pretrained()
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = torch.nn.Conv2d(
        #    self.out_channels, self.out_channels, kernel_size=2, stride=2, padding=1, bias=False
        # )
        # channels = [64, 128, 256, 512]
        dilate = [False, False, False]
        dilation = 1
        inplanes = 32
        self.layer1, dilation, inplanes = self._make_layer(
            channels[0], layers[0], prev_dilation=dilation, inplanes=inplanes
        )
        self.layer2, dilation, inplanes = self._make_layer(
            channels[1], layers[1], stride=2, dilate=dilate[0], prev_dilation=dilation, inplanes=inplanes
        )
        self.layer3, dilation, inplanes = self._make_layer(
            channels[2], layers[2], stride=2, dilate=dilate[1], prev_dilation=dilation, inplanes=inplanes
        )
        self.layer4, dilation, inplanes = self._make_layer(
            channels[3], layers[3], stride=2, dilate=dilate[2], prev_dilation=dilation, inplanes=inplanes
        )

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        prev_dilation: int,
        inplanes: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        # block = Bottleneck
        block = BasicBlock
        norm_layer = nn.BatchNorm2d
        downsample = None
        dilation = prev_dilation
        if dilate:
            dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(inplanes, planes, stride, downsample, self.groups, self.base_width, prev_dilation, norm_layer)
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers), dilation, inplanes

    def set_is_pretrained(self):
        for name, mm in self.named_modules():
            if len([mm for mm in mm.parameters()]) > 0:
                mm.is_pretrained = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetBackboneSmall(ResNetBackboneCustom):
    def __init__(self, in_channels):
        super().__init__(in_channels, layers=[3, 2, 1, 1], channels=[32, 64, 128, 256])

class ResNetBackboneMed(ResNetBackboneCustom):
    def __init__(self, in_channels):
        super().__init__(in_channels, layers=[3, 4, 6, 3], channels=[64, 64, 128, 256])


class ResNetBackboneBig(ResNetBackboneCustom):
    def __init__(self, in_channels):
        super().__init__(in_channels, layers=[3, 4, 6, 3], channels=[64, 128, 256, 512])
