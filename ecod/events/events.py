# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from evis.trans import events_to_frames_2c_fps, events_to_n_frames_2c


def events_to_frames(events, shape, fps=None, n_frames=None, t_start=-1, t_end=-1, binary_spikes=False):
    """

    :param events: events of (t, x, y, p)
    :param shape: (height, width)
    :param fps: Tile events in samples, shuch that each sample is fps long
    :param n_frames: Tile events in n_samples samples
    :param t_start: Start of tiling. If < 0, take first event
    :param t_end: End of tiling. If < 0, take last event
    :return: array(n_frames, 2, shape[0], shape[1])
    """
    if fps is not None:
        return np.array(events_to_frames_2c_fps(events, fps, shape[1], shape[0], t_start, t_end))
    elif n_frames is not None:
        return events_to_n_frames_2c(events, n_frames, shape[1], shape[0], t_start, t_end, binary_spikes)
    raise ValueError("fps and n_frames cannot be both None")


def get_event_fps(slowdown):
    return 60.0 * slowdown


def events_to_frames_slowdown(events, shape, slowdown=1.0, return_last=False, t_start=0.0):
    # ensure enough fps
    fps = get_event_fps(slowdown)
    return events_to_frames(events, shape, fps, return_last=return_last, t_start=t_start)
