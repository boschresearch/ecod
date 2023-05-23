# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from evis.trans import count_events_in_box, count_events_in_boxes

events = np.array([[0.01, 1, 4, 0], [0.1, 4, 5, 1]])
boxes = np.array([[0.0, 1, 10, 6], [5, 10, 20, 15.0]])

x_start = 0
x_stop = 10
y_start = 1
y_stop = 5
t_start_mus = 0.0
t_stop_mus = 0.05

print(count_events_in_box(events, x_start, x_stop, y_start, y_stop, t_start_mus, t_stop_mus))

print(count_events_in_boxes(events, boxes))
