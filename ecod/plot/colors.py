# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _create_ev_quad_colors():
    cmap = plt.get_cmap("PiYG")
    purple = plt.get_cmap("Blues")(0.7)
    # generally, have four colors: off, both, no, on -> -2, -1, 0, 1
    colors = ListedColormap([cmap(0.0), purple, cmap(0.5), cmap(1.0)])
    return colors


EV_QUAD_COLORS = _create_ev_quad_colors()
