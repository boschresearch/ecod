# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
from torchvision.ops import box_convert

from ecod.data.rmmnist.dataset import RandomMoveMnistSingleIndexer, load_rmmnist_events
from ecod.data.box2d.priors import PriorBox, assign_priors, calculate_number_of_boxes_per_feature_map
from ecod.data.box2d.transforms import center_form_to_corner_form
from ecod.data.transforms import VoxelGridTransform
from ecod.events.voxel import events_to_seq_voxel_grid_2c
from ecod.plot.box2d import add_boxes


class SimpleRMMNISTDset:
    def __init__(self, root_path, bbox_suffix, tvt, target_shape, n_bins=5, return_frames=False):
        self.root_path = root_path
        self.bbox_suffix = bbox_suffix
        self.tvt = tvt
        self.target_shape = target_shape
        self.n_bins = n_bins
        self.return_frames = return_frames
        self.indexer = RandomMoveMnistSingleIndexer(root_path, bbox_suffix, tvt)
        self.delta_t_ms = self.indexer.delta_t_ms
        self.seq_transform = VoxelGridTransform(target_shape)

    def load_sample(self, idx):
        file_idx, frame_idx, bbox_start, bbox_end = self.indexer(idx)
        path = self.root_path / self.tvt / f"{file_idx:0>5}.h5"
        with h5py.File(path, "r") as hd:
            bboxs = hd["bboxs_tlxywh"][bbox_start:bbox_end]
            # +1 because of background label == 0
            labels = hd["labels"][bbox_start:bbox_end] + 1
            frame_time = hd["frame_times_ms"][frame_idx]
            if self.return_frames:
                # frame_stop is the last frame index -> +1 to load it
                frame = hd["frames"][frame_idx]
            else:
                frame = None
        return file_idx, frame_time, frame, bboxs.astype(np.float32), labels.astype(int)

    def __getitem__(self, idx):
        file_idx, frame_time, frame, bboxs, labels = self.load_sample(idx)
        frame_times_ms = np.array([frame_time - self.delta_t_ms, frame_time])
        if self.return_frames:
            seq = frame[None, None]
        else:
            events = load_rmmnist_events(self.root_path, file_idx, self.tvt, to_float=True)
            seq = events_to_seq_voxel_grid_2c(
                events, frame_times_ms * 1e3, self.n_bins, self.target_shape[1], self.target_shape[0]
            )
        seq, bboxs, labels = self.seq_transform(seq, bboxs, labels)
        return seq.float(), torch.from_numpy(bboxs).float(), torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.indexer)


def get_gt_seqs_and_boxes(root, bbox_suffix, tvt, target_shape, n_samples, return_frames):
    dset = SimpleRMMNISTDset(root, bbox_suffix, tvt, target_shape, return_frames=return_frames)
    class_names = ["__background__"] + dset.indexer.meta_data["labels"]

    rng = np.random.default_rng(24)
    random_idxs = rng.integers(0, len(dset), size=n_samples)

    frame_col = []
    bboxs_col = []
    labels_col = []
    for idx in random_idxs:
        frame, bboxs, labels = dset[idx]
        frame_col.append(frame.numpy().sum((0, 1)))
        bboxs_col.append(bboxs.clone().numpy())
        labels_col.append(labels.clone().numpy())
    return frame_col, bboxs_col, labels_col, class_names


def assign_priors_to_list_of_bboxs(
    bboxs_col,
    labels_col,
    shape_t,
    prior_feature_maps,
    prior_min_sizes,
    prior_max_sizes,
    prior_strides,
    prior_aspect_ratios,
    target_shape,
    prior_clip=True,
    iou_func="giou",
    iou_thresh=0.5,
    only_best_priors=False,
):
    # define and assign priors
    priors = center_form_to_corner_form(
        PriorBox(
            shape_t=shape_t,
            prior_feature_maps=prior_feature_maps,
            prior_min_sizes=prior_min_sizes,
            prior_max_sizes=prior_max_sizes,
            prior_strides=prior_strides,
            prior_aspect_ratios=prior_aspect_ratios,
            prior_clip=prior_clip,
            debug_mode=False,
        )()
    )

    priors_col = []
    for idx in range(len(bboxs_col)):
        bboxs = torch.from_numpy(bboxs_col[idx].copy())
        labels = torch.from_numpy(labels_col[idx].copy())
        bboxs[:, ::2] /= target_shape[1]
        bboxs[:, 1::2] /= target_shape[0]
        bboxs = box_convert(bboxs, "xywh", "xyxy")
        assigned_priors, labels = assign_priors(
            bboxs, labels, priors, iou_thresh, only_best_priors, debug_mode=True, iou_func=iou_func
        )
        assigned_priors = assigned_priors[labels > 0]
        assigned_priors[:, ::2] *= target_shape[1]
        assigned_priors[:, 1::2] *= target_shape[0]
        assigned_priors = np.c_[labels[labels > 0], assigned_priors]
        priors_col.append(assigned_priors)
    return priors_col


def plot_assigned_priors(frame_col, bboxs_col, labels_col, priors_col, class_names):
    n_samples = len(frame_col)
    n_cols = int(np.sqrt(n_samples))
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(n_cols, n_rows, figsize=(2.0 * n_rows, 2.0 * n_cols))
    if type(axes) is not np.ndarray:
        axes = np.array([axes]).reshape(1, -1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    n_labels = len(class_names)
    for ii in range(len(frame_col)):
        frame = frame_col[ii]
        bboxs = bboxs_col[ii]
        labels = labels_col[ii]
        priors_ass = priors_col[ii]
        ax = axes.flat[ii]
        ax.imshow(frame)
        bboxs_plot = np.c_[labels, bboxs]
        add_boxes(
            bboxs_plot,
            n_labels,
            ax,
            text=True,
            names=class_names,
            loc="topleft",
            lw_scale=2.0,
            box_kwargs=None,
            cmap="Dark2",
        )
        add_boxes(
            priors_ass,
            n_labels,
            ax,
            text=False,
            names=class_names,
            loc="tlbr",
            lw_scale=2.0,
            box_kwargs=None,
            cmap="plasma",
        )
    return fig, axes


def plot_priors_per_feature_map(
    shape_t,
    prior_feature_maps,
    prior_min_sizes,
    prior_max_sizes,
    prior_strides,
    prior_aspect_ratios,
    prior_clip=True,
    frame=None,
):
    prior_boxes = PriorBox(
        shape_t=shape_t,
        prior_feature_maps=prior_feature_maps,
        prior_min_sizes=prior_min_sizes,
        prior_max_sizes=prior_max_sizes,
        prior_strides=prior_strides,
        prior_aspect_ratios=prior_aspect_ratios,
        prior_clip=prior_clip,
        debug_mode=True,
    )()
    n_boxes_at_map = calculate_number_of_boxes_per_feature_map(prior_aspect_ratios, prior_max_sizes)

    fig, axes = plt.subplots(2, len(prior_boxes) + 1, figsize=(10, 5))
    if frame is None:
        frame = np.ones(shape_t[-2:])
    axes[0, -1].imshow(frame)
    axes[1, -1].imshow(frame)
    target_shape = shape_t[-2:]
    for ii, prior_boxes_per_map in enumerate(prior_boxes):
        axes[0, ii].imshow(frame)
        # select middle set of boxes
        box_mid = prior_boxes_per_map[int(len(prior_boxes_per_map)//2)]
        mask = (prior_boxes_per_map[:, 0] == box_mid[0]) & (prior_boxes_per_map[:, 1] == box_mid[1])
        bboxs = prior_boxes_per_map[mask]
        bboxs = np.c_[ii * np.ones(bboxs.shape[:1]), bboxs]
        bboxs[:, 1::2] *= target_shape[0]
        bboxs[:, 2::2] *= target_shape[1]
        add_boxes(
            bboxs,
            len(prior_boxes),
            axes[0, ii],
            text=False,
            names=None,
            loc="center",
            lw_scale=2.0,
            box_kwargs=None,
            cmap="Dark2",
        )
        add_boxes(
            bboxs,
            len(prior_boxes),
            axes[0, -1],
            text=False,
            names=None,
            loc="center",
            lw_scale=2.0,
            box_kwargs=None,
            cmap="Dark2",
        )
        # lower part of figure
        axes[1, ii].imshow(frame)
        centers = prior_boxes_per_map[:, :2].clone()
        centers[:, 1] *= target_shape[0]
        centers[:, 0] *= target_shape[1]
        axes[1, ii].plot(centers[:, 0], centers[:, 1], ".", markersize=10)
        axes[1, -1].plot(centers[:, 0], centers[:, 1], ".", markersize=10)
    return fig, axes
