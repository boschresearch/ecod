# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import numpy as np
import h5py
import torch
from torch import from_numpy

from numba import njit, prange

from torchvision.ops import box_iou, box_area, batched_nms

from evis.trans import count_events_in_box, count_events_in_boxes

from ecod.plot.boxmemory import BoxMemoryPlotter, rescale_boxes


def count_events_in_box(events, x_start, x_stop, y_start, y_stop, t_start_mus, t_stop_mus):
    mask = np.full((len(events),), True, dtype=bool)
    for idx, val in zip([0, 1, 2], [t_start_mus, x_start, y_start]):
        mask_this = events[:, idx] > val
        mask = mask & mask_this
    for idx, val in zip([0, 1, 2], [t_stop_mus, x_stop, y_stop]):
        mask_this = events[:, idx] < val
        mask = mask & mask_this
    return mask.astype(int).sum()


@njit(nogil=True, parallel=True)
def optimized_count(events, boxes, masks):
    for ii in prange(len(boxes)):
        box = boxes[ii]
        for idx, val in zip([1, 2], [box[0], box[1]]):
            mask_this = events[:, idx] > val
            masks[ii] = masks[ii] & mask_this
        for idx, val in zip([1, 2], [box[2], box[3]]):
            mask_this = events[:, idx] < val
            masks[ii] = masks[ii] & mask_this
    return None


def count_events_in_boxes2(events, boxes):
    masks = np.full((len(boxes), len(events)), True)
    optimized_count(events, boxes, masks)
    return masks.astype(int).sum(1)


def count_events_in_boxes3(events, boxes):
    counts = []
    for box in boxes:
        counts.append(count_events_in_box(events, box[0], box[2], box[1], box[3], -1, np.inf))
    return np.array(counts)


class BoxMemory:
    """Bounding box memory based on events

    Basic principle:
        1. Memorize all bounding boxes at time step t with score >= conf_thresh
        2. At t+1, count events for all memorized boxes
            If >= forget_thresh, forget box because object most likely moved
            If < forget_thresh, append box to regular predictions, because few events mean very little movement
        3. Reject box predictions at t+1 with n_evs < delete_thresh

    """

    def __init__(
        self,
        hparams,
        events_shape,
        box_shape,
        conf_thresh=0.3,
        forget_thresh=0.0001,
        delete_thresh=100,
        iou_thresh=0.5,
        plot=False,
        savepath=None,
        load_counts=False,
    ):
        self.hparams = hparams
        self.events_shape = events_shape
        self.box_shape = box_shape
        self.conf_thresh = conf_thresh
        self.forget_thresh = forget_thresh
        self.delete_thresh = delete_thresh
        self.delete_use_areas = 0 <= self.delete_thresh < 1.0
        self.forget_use_areas = 0 <= self.forget_thresh < 1.0
        self.iou_thresh = iou_thresh
        self.plot = plot
        self.savepath = savepath
        self.do_load_counts = load_counts
        self.do_save_counts = False
        self.scale_factors_xy = [events_shape[ii] / box_shape[ii] for ii in range(2)][::-1]
        self.box_mem = None
        self.file_idx = -1
        self._internal_idx = 0
        self.stats = {
            "forget": 0,
            "forget_close": 0,
            "forget_count": 0,
            "remember": 0,
            "add": 0,
            "resets": 0,
            "delete": 0,
            "add_iou": 0,
            "idx": 0,
            "actual_add": 0,
        }
        self.mem_idxs = None
        self.reset_mem_idxs()
        self.clear_memory()
        if self.plot:
            self.debug_plotter = BoxMemoryPlotter(events_shape, box_shape, conf_thresh, savepath)
        else:
            self.debug_plotter = None

    def reset_mem_idxs(self):
        self.mem_idxs = {
            "forget": np.zeros((0,), dtype=int),
            "add_iou": np.zeros((0,), dtype=int),
            "add_count": np.zeros((0,), dtype=int),
        }

    def clear_memory(self):
        self.box_mem = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    def get_counts_savepath(self):
        return Path(self.savepath) / "counts.h5"

    def get_counts_mask(self, events):
        boxes_scaled = rescale_boxes(self.box_mem["boxes"], self.scale_factors_xy)
        counts = count_events_in_boxes(events, boxes_scaled)
        if self.forget_use_areas:
            areas = box_area(torch.from_numpy(boxes_scaled)).numpy()
            mask = counts / areas >= self.forget_thresh
        else:
            mask = counts >= self.forget_thresh
        return mask

    def index_memory(self, boxes, labels, scores, events):
        if len(self.box_mem["boxes"]) == 0:
            return None
        mask = self.get_counts_mask(events)
        # case event_count >= forget_thresh
        idxs = np.nonzero(mask)[0]
        if self.iou_thresh > 0.0:
            mask_scores = scores >= self.conf_thresh
            good_boxes = boxes[mask_scores]
            if len(good_boxes) == 0:
                self.mem_idxs["add_iou"] = idxs
            else:
                mem_boxes = self.box_mem["boxes"][idxs]
                ious = box_iou(torch.from_numpy(good_boxes), torch.from_numpy(mem_boxes)).numpy().max(0)
                mask_ious = ious < self.iou_thresh
                self.mem_idxs["add_iou"] = np.nonzero(mask_ious)[0]
                self.mem_idxs["forget"] = np.nonzero(~mask_ious)[0]
        else:
            self.mem_idxs["forget"] = idxs
        # case event_count < forget_thresh
        self.mem_idxs["add_count"] = np.nonzero(~mask)[0]
        return None

    def nms_after_adding_from_memory(self, boxes, labels, scores):
        """This can help if boxes are added and also a prediction was made

        Args:
            boxes ([type]): [description]
            labels ([type]): [description]
            scores ([type]): [description]

        Returns:
            [type]: Indices of boxes to keep
        """
        keep_idxs = batched_nms(
            from_numpy(boxes),
            from_numpy(scores),
            from_numpy(labels),
            self.hparams["test_nms_threshold"],
        ).numpy()
        return keep_idxs

    def filter_boxes_from_memory_close_to_predictions(self, boxes, scores, mem_boxes):
        # TODO: Maybe do separate per class
        IOU_THRESH_MEM_CLOSE_PRED = 0.5
        mask = scores >= self.conf_thresh
        good_boxes = boxes[mask]
        if len(good_boxes) > 0:
            ious = box_iou(torch.from_numpy(good_boxes), torch.from_numpy(mem_boxes)).numpy().max(0)
            keep_idxs = np.nonzero(ious < IOU_THRESH_MEM_CLOSE_PRED)[0]
        else:
            keep_idxs = None
        return keep_idxs

    def filter_close_boxes_from_memory(self, new_boxes):
        # TODO: Maybe do separate per class
        IOU_THRESH_MEM_CLOSE_MEM = 0.5
        mem_boxes = self.box_mem["boxes"]
        if len(mem_boxes) > 0 and len(new_boxes) > 0:
            mem_torch = torch.from_numpy(mem_boxes)
            new_torch = torch.from_numpy(new_boxes)
            ious = box_iou(new_torch, mem_torch).numpy()
            ious_max = ious.max(0)
            keep_idxs = np.nonzero(ious_max < IOU_THRESH_MEM_CLOSE_MEM)[0]
        else:
            keep_idxs = None
        return keep_idxs

    def add_boxes_from_memory(self, boxes, labels, scores):
        idxs_count = self.mem_idxs["add_count"]
        idxs_iou = self.mem_idxs["add_iou"]
        idxs = np.concatenate([idxs_count, idxs_iou], axis=0)
        if len(idxs) > 0:
            mem_boxes = self.box_mem["boxes"][idxs]
            mem_labels = self.box_mem["labels"][idxs]
            mem_scores = self.box_mem["scores"][idxs]
            keep_idxs = self.filter_boxes_from_memory_close_to_predictions(boxes, scores, mem_boxes)
            if keep_idxs is not None:
                mem_boxes, mem_labels, mem_scores = (
                    mem_boxes[keep_idxs],
                    mem_labels[keep_idxs],
                    mem_scores[keep_idxs],
                )
            self.stats["actual_add"] += len(mem_labels)
            boxes = np.concatenate([boxes, mem_boxes], axis=0)
            labels = np.concatenate([labels, mem_labels], axis=0)
            scores = np.concatenate([scores, mem_scores], axis=0)
            # keep_idxs = self.nms_after_adding_from_memory(boxes, labels, scores)
            # boxes, labels, scores = boxes[keep_idxs], labels[keep_idxs], scores[keep_idxs]
        self.stats["add"] += len(idxs_count)
        self.stats["add_iou"] += len(idxs_iou)
        return boxes, labels, scores

    def forget_boxes(self):
        # reverse indices, otherwise with each pop the indices would change
        for key in ["boxes", "labels", "scores"]:
            self.box_mem[key] = np.delete(self.box_mem[key], self.mem_idxs["forget"], axis=0)
        self.stats["forget"] += len(self.mem_idxs["forget"])
        self.stats["forget_count"] += len(self.mem_idxs["forget"])

    def add_boxes_to_memory(self, boxes, labels, scores):
        mask = scores >= self.conf_thresh
        good_boxes, good_labels, good_scores = boxes[mask], labels[mask], scores[mask]
        keep_idxs = self.filter_close_boxes_from_memory(good_boxes)
        for key, val in zip(["boxes", "labels", "scores"], [good_boxes, good_labels, good_scores]):
            if keep_idxs is not None:
                self.stats["forget_close"] += len(keep_idxs)
                self.stats["forget"] += len(keep_idxs)
                self.box_mem[key] = self.box_mem[key][keep_idxs]
            self.box_mem[key] = np.concatenate([self.box_mem[key], val], axis=0)
        self.stats["remember"] += len(good_boxes)

    def load_counts(self, n_counts):
        path = self.get_counts_savepath()
        with h5py.File(path, "a") as hd:
            counts = np.array(hd[f"{self._internal_idx:0>8}"], dtype=int)
        if len(counts) != n_counts:
            raise RuntimeError(
                "Length of loaded counts and predicted boxes has to match, but are: " f"{len(counts)}, {n_counts}"
            )
        return counts

    def save_counts(self, counts):
        path = self.get_counts_savepath()
        if path.exists():
            path.unlink()
        with h5py.File(path, "a") as hd:
            hd.create_dataset(f"{self._internal_idx:0>8}", data=counts, dtype=np.uint32)

    def delete_boxes_with_low_event_count(self, boxes, labels, scores, events):
        boxes_scaled = rescale_boxes(boxes, self.scale_factors_xy)
        if self.do_load_counts:
            counts = self.load_counts(len(boxes_scaled))
        else:
            counts = count_events_in_boxes(events, boxes_scaled)
            if self.savepath is not None and self.do_save_counts:
                self.save_counts(counts)
        if self.delete_use_areas:
            areas = box_area(torch.from_numpy(boxes_scaled)).numpy()
            mask = counts / areas < self.delete_thresh
        else:
            mask = counts < self.delete_thresh
        to_delete = np.nonzero(mask)[0]
        self.stats["delete"] += len(to_delete)
        boxes = np.delete(boxes, to_delete, axis=0)
        labels = np.delete(labels, to_delete, axis=0)
        scores = np.delete(scores, to_delete, axis=0)
        self._internal_idx += 1
        return boxes, labels, scores

    def __call__(self, boxes, labels, scores, events, file_idx, boxes_gt=None, labels_gt=None):
        plot = self.plot  # and self.file_idx == 4
        if plot:
            self.debug_plotter.plot_boxes_before_memory(boxes, labels, scores, events, file_idx, boxes_gt, labels_gt)
        if self.file_idx != file_idx:
            self.clear_memory()
            self.file_idx = file_idx
            self.stats["resets"] += 1
            self.stats["idx"] = 0
        boxes, labels, scores = self.delete_boxes_with_low_event_count(boxes, labels, scores, events)
        self.index_memory(boxes, labels, scores, events)
        boxes_w_mem, labels_w_mem, scores_w_mem = self.add_boxes_from_memory(boxes, labels, scores)
        self.forget_boxes()
        self.add_boxes_to_memory(boxes, labels, scores)
        self.stats["idx"] += 1
        if plot:
            self.debug_plotter.plot_boxes_after_memory(boxes_w_mem, labels_w_mem, scores_w_mem, events, file_idx)
        self.reset_mem_idxs()
        return boxes_w_mem, labels_w_mem, scores_w_mem
