# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_grid(size, shape, dpi=100.0, scale=10.0, gss_kwargs=None):
    gss_kwargs_defaults = dict(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    if gss_kwargs is not None:
        gss_kwargs_defaults.update(gss_kwargs)
    size_sq = np.sqrt(size)
    if size_sq > int(size_sq):
        gridshape = (int(size_sq) + 1, int(size_sq) + 1)
        n_tot = gridshape[0] * gridshape[1]
        n_cols = gridshape[0]
        while (n_tot - gridshape[1]) >= size:
            n_tot -= gridshape[1]
            n_cols -= 1
        gridshape = (n_cols, gridshape[1])
    else:
        gridshape = (int(size_sq), int(size_sq))
    width = shape[1] * gridshape[1] / dpi * scale
    height = shape[0] * gridshape[0] / dpi * scale
    fig = plt.figure(figsize=(width, height))
    gss = gridspec.GridSpec(gridshape[0], gridshape[1])
    # set the spacing between axes
    gss.update(**gss_kwargs_defaults)
    # only define axes that are needed
    axes = np.array([plt.subplot(gss[ii]) for ii in range(size)])
    for ax in axes:
        ax.axis("off")
    return fig, axes, gss
