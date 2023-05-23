# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np


def get_shape_from_events(events):
    shape = (events[:, 1:3].max(0) + 1).astype(int)
    # events are x,y but shape is y,x, therefore [1], [0]
    return np.array([shape[1], shape[0]])


def crop_events_center(events, crop_shape, events_shape=None):
    crop_shape = np.array(crop_shape)
    if events_shape is None:
        events_shape = get_shape_from_events(events)
    center = events_shape // 2
    yx_start = center - crop_shape // 2
    yx_end = center + crop_shape // 2
    mask_x = np.logical_and(events[:, 1] >= yx_start[1], events[:, 1] < yx_end[1])
    mask_y = np.logical_and(events[:, 2] >= yx_start[0], events[:, 2] < yx_end[0])
    events_c = events[mask_x & mask_y]
    events_p = events_c.copy()
    # shift events to center
    for idx, pad in zip([2, 1], yx_start):
        events_p[:, idx] -= pad
    return events_p


class RandomShiftEventsXY:
    def __init__(self, width, height, x_max=5, y_max=5, seed=211):
        # width and height have to be BEFORE reshaping
        self.width = width
        self.height = height
        self.x_max = x_max
        self.y_max = y_max
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, events, boxes=None, labels=None):
        # need to copy?
        events_c = events
        if self.x_max > 0:
            x_s = self.rng.randint(2 * self.x_max + 1) - self.x_max
            events_c[:, 1] += x_s
            events_c = events_c[events_c[:, 1] < self.width]
            events_c = events_c[events_c[:, 1] >= 0]
            if boxes is not None:
                for ii in [0, 2]:
                    boxes[:, ii] += x_s
                    # boxes = boxes[boxes[:, ii]<self.width]
                    # boxes = boxes[boxes[:, ii]>=0]
                    # small boxes are filtered anyway so don't have to filter here
                    boxes[boxes[:, ii] >= self.width, ii] = self.width - 1
                    boxes[boxes[:, ii] < 0, ii] = 0
        if self.y_max > 0:
            y_s = self.rng.randint(2 * self.y_max + 1) - self.y_max
            events_c[:, 2] += y_s
            events_c = events_c[events_c[:, 2] < self.height]
            events_c = events_c[events_c[:, 2] >= 0]
            if boxes is not None:
                for ii in [1, 3]:
                    boxes[:, ii] += y_s
                    # boxes = boxes[boxes[:, ii]<self.height]
                    # boxes = boxes[boxes[:, ii]>=0]
                    # small boxes are filtered anyway so don't have to filter here
                    boxes[boxes[:, ii] >= self.height, ii] = self.height - 1
                    boxes[boxes[:, ii] < 0, ii] = 0
        return events_c, boxes, labels


class ShiftEventsTime:
    def __init__(self, t_max=2, seed=211):
        self.t_max = t_max
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, events, boxes=None, labels=None):
        # need to copy events?
        if self.t_max > 0:
            t_s = self.rng.randint(2 * self.t_max + 1) - self.t_max
            events[:, 0] += t_s
            # events could also start at arbitrary point...is it a problem when event times are negative?
            events = events[events[:, 0] > 0]
        return events, boxes, labels


# not yet finished
class AddRandomEvents:
    def __init__(self, width, height, area_percentage=0.02, rate_max=10.0, seed=2561):
        self.width = width
        self.height = height
        self.seed = seed
        self.area_percentage = area_percentage
        # Hz
        self.rate_max = rate_max
        self.rng = np.random.RandomState(seed)
        self.n_pixels = self.width * self.height
        self.n_pixels_sample = int(self.n_pixels * self.area_percentage)
        self.grid = np.mgrid[0:height, 0:width].reshape(2, -1).T

    def __call__(self, events, boxes, labels):
        pixel_idxs = self.rng.randint(0, self.n_pixels, size=self.n_pixels_sample)
        for idx in pixel_idxs:
            y, x = self.grid[idx]
            delta_t = events[-1, 0] - events[0, 0]
            ts = self.rng.exponential(1 / self.rate_max, size=10000)


class RemoveRandomEvents:
    def __init__(self, remove_prob=0.05, seed=5181):
        self.seed = seed
        self.remove_prob = remove_prob
        self.rng = np.random.RandomState(seed)

    def __call__(self, events, boxes, labels):
        mask = self.rng.binomial(1, 1 - self.remove_prob, size=len(events)).astype(bool)
        return events[mask], boxes, labels


class SortEvents:
    def __init__(self):
        pass

    def __call__(self, events, boxes, labels):
        argsort = np.argsort(events[:, 0])
        return events[argsort], boxes, labels


class EventTransforms:
    def __init__(self, width, height, x_max, y_max, t_max, remove_prob):
        self.shift_xy = ShiftEventsXY(width, height, x_max, y_max)
        self.shift_t = ShiftEventsTime(t_max)
        self.remove_events = RemoveRandomEvents(remove_prob=remove_prob)

    def __call__(self, events, boxes, labels):
        # for some reason, the loop like in the torchvision.transforms.Compose implemenation does lead to a segfault,
        # so just do it 'manually'
        return self.shift_xy(*self.shift_t(*self.remove_events(events, boxes, labels)))
