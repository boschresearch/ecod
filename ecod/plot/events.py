# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from ecod.plot.colors import EV_QUAD_COLORS
from ecod.plot.general import get_grid


def ev_frames_to_quad(frames):
    frames = frames.copy()
    frames[frames > 1] = 1
    frames = frames[:, 0] - 2 * frames[:, 1]
    return frames


def event_frames_to_grayscale(event_frames):
    frames_diff = event_frames[:, 0] - event_frames[:, 1]
    # frames_diff -= frames_diff.min()
    # frames_diff /= np.percentile(frames_diff, 50)
    return frames_diff


def plot_frames(frames, dpi=100.0, scale=5.0, frame_type="gs", **kwargs):
    height, width = frames.shape[-2:]
    shape = (height, width)
    fig, axes, gss = get_grid(len(frames), shape, dpi, scale)
    imshow_frames(frames, axes, frame_type, **kwargs)
    gss.tight_layout(fig, pad=0.01, h_pad=0.01, w_pad=0.01)
    for ax in axes.flat:
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def imshow_frames(frames, axes, frame_type, **kwargs):
    if frame_type == "gs":
        f_min = frames.min()
        f_max = frames.max()
        for ii, frame in enumerate(frames):
            imshow_gs(frame, axes.flat[ii], vmin=f_min, vmax=f_max, **kwargs)
    elif frame_type == "ev":
        frames = ev_frames_to_quad(frames)
        for ii, frame in enumerate(frames):
            imshow_ev_quad(frame, axes.flat[ii], **kwargs)
    else:
        raise ValueError("frame_type in 'ev', 'gs'")


def imshow_gs(frame, ax, im=None, cmap="gray", **kwargs):
    if im is None:
        im = ax.imshow(frame, cmap=cmap, **kwargs)
    else:
        im.set_array(frame)
    return im


def imshow_ev_quad(frame, ax, im=None, **kwargs):
    if im is None:
        im = ax.imshow(frame, cmap=EV_QUAD_COLORS, vmin=-2, vmax=1, **kwargs)
    else:
        im.set_array(frame)
    return im
