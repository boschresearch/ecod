# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


def trim_events(events, shape):
    mask1 = events[:, 1] < shape[1]
    mask2 = events[:, 2] < shape[0]
    events_m = events[mask1 & mask2]
    return events_m
