# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

#!usr/bin/env python

import numpy as np

from evis.trans import events_to_n_frames_2c, events_to_frames_2c_fps


def main():
    events = np.array(
        [
            [0.0, 2.0, 3.0, 1.0],
            [10000.0, 0.0, 0.0, 1.0],
            [30000.0, 0.0, 5.0, -1.0],
            [60000.0, 0.0, 5.0, -1.0],
            [70000.0, 1.0, 9.0, -1.0],
        ]
    )
    events[:, 0] *= 1e3

    shape = (20, 30)
    n_frames = 10
    for t_end in [1e6, 10e6, 10.000001e6, 9.99999999e6, 80e6, 80.14e6]:
        for t_start in [0, 12e6]:
            if t_start >= t_end:
                continue
            print("T: ", t_start, t_end)
            fps = n_frames / ((t_end - t_start) / 1e6)
            frames = np.array(events_to_frames_2c_fps(events, fps, shape[0], shape[1], t_start, t_end))
            fsum = frames.sum((1, 2, 3))
            print("FPS", frames.shape, fsum)
            frames2 = events_to_n_frames_2c(events, n_frames, shape[0], shape[1], t_start, t_end, False)
            print(f"dtype n_frames: {frames2.dtype}")
            fsum2 = frames2.sum((1, 2, 3))
            print("N", frames2.shape, fsum2)
            if (fsum != fsum2).any():
                print("WARN SOMETHING IS WRONG")
            print()


if __name__ == "__main__":
    main()
