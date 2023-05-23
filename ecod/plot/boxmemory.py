# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from torch import from_numpy
from torchvision.ops import box_area

from evis.trans import count_events_in_boxes

from ecod.utils.files import makedirs
from ecod.plot.box2d import add_boxes
from ecod.plot.events import ev_frames_to_quad, imshow_ev_quad


def events_to_binary_image(events, events_shape):
    image = np.zeros(events_shape)
    image[events[:, 2].astype(int), events[:, 1].astype(int)] = 1
    return image


def events_to_binary_image_2c(events, events_shape):
    image = np.zeros((2, *events_shape))
    mask_pos = events[:, 3] == 1
    events_pos = events[mask_pos]
    mask_neg = ~mask_pos
    events_neg = events[mask_neg]
    image[0, events_pos[:, 2].astype(int), events_pos[:, 1].astype(int)] = 1
    image[1, events_neg[:, 2].astype(int), events_neg[:, 1].astype(int)] = 1
    return image


def rescale_boxes(boxes, scale_factors_xy):
    boxes = boxes.copy()
    boxes[:, 0::2] *= scale_factors_xy[0]
    boxes[:, 1::2] *= scale_factors_xy[1]
    return boxes


class BoxMemoryPlotter:
    def __init__(self, events_shape, box_shape, conf_thresh, savepath):
        self.events_shape = events_shape
        self.box_shape = box_shape
        self.conf_thresh = conf_thresh
        self.scale_factors_xy = [events_shape[ii] / box_shape[ii] for ii in range(2)][::-1]
        self.cur_image = 0
        self.savepath = Path(savepath) / "box_memory_plots"
        self.gt_in_same = True
        self.add_text = True
        self.filter_for_plot = False
        makedirs(self.savepath, overwrite=True)
        if self.gt_in_same:
            self.fig, self.axes = plt.subplots(2, 1, figsize=(6.0, 6.75), constrained_layout=True)
        else:
            self.fig, self.axes = plt.subplots(3, 1, figsize=(6.0, 10.125), constrained_layout=True)
        self.reset()

    def reset(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.axis("off")

    def add_box_text(self, box, score, count, area_count, ax):
        if self.add_text:
            ax.text(
                box[1],
                box[2],
                f"{score:.2f}/{count}/{area_count:.2g}",
                color="black",
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"),
            )

    def events_to_frame(self, events):
        ev_frame = events_to_binary_image_2c(events, self.events_shape)
        frame = ev_frames_to_quad(ev_frame[None])[0]
        return frame

    def add_boxes(self, boxes, ax, cmap="jet", lw_scale=1.0):
        # hard-coded n_classes here; change if it should be higher
        add_boxes(boxes, 5, ax, text=False, loc="tlbr", cmap=cmap, lw_scale=lw_scale)

    def _filter_boxes(self, boxes, labels, scores=None):
        if self.filter_for_plot:
            mask = boxes[:, 0] < 10
            boxes, labels = boxes[mask], labels[mask]
            if scores is not None:
                scores = scores[mask]
        return boxes, labels, scores

    def plot_boxes_before_memory(self, boxes, labels, scores, events, file_idx, boxes_gt=None, labels_gt=None):
        ax1 = self.axes.flat[0]
        # image = events_to_binary_image(events, self.events_shape)
        image = self.events_to_frame(events)
        for ax in self.axes.flat:
            # ax.imshow(image)
            imshow_ev_quad(image, ax)
        boxes = rescale_boxes(boxes, self.scale_factors_xy)
        mask = scores >= self.conf_thresh
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        boxes, labels, scores = self._filter_boxes(boxes, labels, scores)
        counts = count_events_in_boxes(events, boxes)
        areas = box_area(from_numpy(boxes)).numpy()
        area_counts = counts / areas
        boxes = np.c_[labels, boxes]
        self.add_boxes(boxes, ax1, cmap="jet")
        for box, score, count, area_count in zip(boxes, scores, counts, area_counts):
            self.add_box_text(box, score, count, area_count, ax1)
        if boxes_gt is not None:
            boxes_gt = rescale_boxes(boxes_gt, self.scale_factors_xy)
            boxes_gt, labels_gt, _ = self._filter_boxes(boxes_gt, labels_gt)
            boxes_gt = np.c_[labels_gt, boxes_gt]
            if self.gt_in_same:
                for ax in self.axes:
                    self.add_boxes(boxes_gt, ax, cmap="Set3", lw_scale=2.0)
            else:
                ax3 = self.axes.flat[2]
                self.add_boxes(boxes_gt, ax3, cmap="jet")

    def plot_boxes_after_memory(self, boxes, labels, scores, events, file_idx):
        boxes = rescale_boxes(boxes, self.scale_factors_xy)
        mask = scores >= self.conf_thresh
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        boxes, labels, scores = self._filter_boxes(boxes, labels, scores)
        counts = count_events_in_boxes(events, boxes)
        areas = box_area(from_numpy(boxes)).numpy()
        boxes = np.c_[labels, boxes]
        area_counts = counts / areas
        ax2 = self.axes.flat[1]
        self.add_boxes(boxes, ax2)
        for box, score, count, area_count in zip(boxes, scores, counts, area_counts):
            self.add_box_text(box, score, count, area_count, ax2)
        self.fig.savefig(self.savepath / f"{file_idx}_{self.cur_image:0>4}_box_mem.png")
        self.reset()
        self.cur_image += 1
