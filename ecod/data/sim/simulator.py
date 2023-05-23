# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d

from evis.simulator import EventSimulator, Config

from ecod.utils.general import ProgressLogger


def events_np_to_float(events):
    return np.c_[events["t"].astype(np.float64) / 1e3, events["x"], events["y"], events["p"]]


class EventsToArray:
    def __init__(self, expected_shape=100000):
        dtype = np.dtype([("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", "?")])
        self.array = np.empty((expected_shape,), dtype=dtype)
        self.cur_idx = 0
        self.expected_shape = expected_shape

    def add_events(self, events):
        time = events.getTime()
        x = events.getx()
        y = events.gety()
        pol = events.getPol()
        n_events = len(time)
        while self.cur_idx + n_events > self.array.shape[0]:
            self.array.resize((self.array.shape[0] + self.expected_shape,))
        self.array["t"][self.cur_idx : self.cur_idx + n_events] = time
        self.array["x"][self.cur_idx : self.cur_idx + n_events] = x
        self.array["y"][self.cur_idx : self.cur_idx + n_events] = y
        self.array["p"][self.cur_idx : self.cur_idx + n_events] = pol
        self.cur_idx += n_events

    def return_array(self):
        return self.array[: self.cur_idx].copy()


def simulate_events(frames, config, fps, print_progress=True):
    esim = EventSimulator(config)
    event_to_array = EventsToArray(int(1e6))
    time_ns = 0
    if print_progress:
        it = ProgressLogger(frames, name="SIM")
    else:
        it = frames
    for frame in it:
        events_now = esim.simulate(frame, time_ns, int(1e9 / fps))
        event_to_array.add_events(events_now)
        time_ns += int(1e9 / fps)
    return event_to_array.return_array()


def process_frames(frames):
    return gaussian_filter1d(frames, 10.0, axis=0)


def get_config(cp=0.6, cm=0.6, sigma_cp=0.0, sigma_cm=0.0, ref_period_ns=int(1e6), log_eps=0.001):
    config = Config()
    config.Cp = cp
    config.Cm = cm
    config.sigma_Cp = sigma_cp
    config.sigma_Cm = sigma_cm
    config.refractory_period_ns = ref_period_ns
    config.log_eps = log_eps
    return config


def save_events(events, savedir, idx, compress=True, use_numpy=True):
    if use_numpy:
        savepath = savedir / f"{idx:0>5}_events.npy"
        np.save(savepath, events, allow_pickle=False)
    else:
        savepath = savedir / f"{idx:0>5}_events.h5"
        with h5py.File(savepath, "w") as hd:
            compression = "lzf" if compress else None
            hd.create_dataset("events", data=events, compression=compression)
    return savepath
