# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch


def get_output_shape(layer, input_shape, device="cpu"):
    with torch.no_grad():
        if hasattr(layer, "get_output_shape"):
            output_shape = layer.get_output_shape(input_shape)
        else:
            output_shape = _get_output_shape(layer, input_shape, device)
    return output_shape


def _get_output_shape(layer, input_shape, device="cpu"):
    with torch.no_grad():
        input_shape = [1] + list(input_shape)
        out = layer.to(device)(torch.zeros(input_shape, device=device))
        output_shape = list(out.shape[1:])
    return output_shape
