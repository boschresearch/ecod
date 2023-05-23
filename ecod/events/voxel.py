"""Build events from voxel grids.

events_to_voxel_grid_2c function adapted from
https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py#L431
licensed under GNU General Public License v3.0
"""
# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from numba import njit, float32, void, int64, prange

import torch
from torch.nn import AdaptiveAvgPool2d


@njit(void(float32[:], int64[:], float32[:]), nogil=True)
def add_at(array_1d, indices, values):
    for ii in prange(len(indices)):
        array_1d[indices[ii]] += values[ii]


@njit
def events_to_voxel_grid_2c(events, num_bins, width, height, t_start=None, t_end=None):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    No interpolation of x and y takes place; it is implicitly padded with zeros on the lower right.
    """
    voxel_grid = np.zeros((num_bins * 2 * height * width,), np.float32)

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0] if t_end is None else t_end
    first_stamp = events[0, 0] if t_start is None else t_start
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].astype(np.int64)
    pols[pols == -1] = 0  # polarity has to be 0 / 1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = 1.0 - dts
    vals_right = dts

    valid_indices = tis < num_bins
    add_at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + pols[valid_indices] * width * height
        + tis[valid_indices] * width * height * 2,
        vals_left[valid_indices].astype(np.float32),
    )

    valid_indices = (tis + 1) < num_bins
    add_at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * width
        + pols[valid_indices] * width * height
        + (tis[valid_indices] + 1) * width * height * 2,
        vals_right[valid_indices].astype(np.float32),
    )

    return np.reshape(voxel_grid, (num_bins, 2, height, width))


def events_to_seq_voxel_grid_2c(events, times_mus, num_bins, width, height):
    """
    events: [t (microseconds), x, y, p]
    No interpolation of x and y takes place; it is implicitly padded with zeros on the lower right.
    """
    n_timesteps = len(times_mus) - 1
    starts = np.searchsorted(events[:, 0], times_mus)
    grids = np.empty((n_timesteps, num_bins, 2, height, width), dtype=np.float32)
    for ii in range(n_timesteps):
        t_start, t_end = times_mus[ii : ii + 2]
        start, stop = starts[ii : ii + 2]
        voxel_grid = events_to_voxel_grid_2c(events[start:stop], num_bins, width, height, t_start=t_start, t_end=t_end)
        grids[ii] = voxel_grid
    return grids


class VoxelGridResizerAvgPool:
    def __init__(self, shape):
        self.shape = shape
        self.aa_pool = AdaptiveAvgPool2d(shape)

    @torch.no_grad()
    def __call__(self, grids):
        org_shape = grids.shape
        gg = grids.reshape(-1, 1, *grids.shape[-2:])
        resized = self.aa_pool(torch.from_numpy(gg))
        return resized.reshape(*org_shape[:-2], *resized.shape[-2:]).numpy()


def resize_voxel_grid_sk(grids, shape):
    """This is about a factor of 100(!) slower than the torch version (270 ms vs 3.3 ms), and this version is
    already 5 times faster than without reshaping (1.2 s vs 270 ms).
    """
    from skimage.transform import resize

    org_shape = grids.shape
    gg = grids.reshape(-1, *grids.shape[-2:])
    resized = resize(gg, [gg.shape[0], 300, 300])
    return resized.reshape(*org_shape[:-2], *resized.shape[-2:])
