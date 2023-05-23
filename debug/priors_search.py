# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Script to do a random search over the prior parameters.

Goal of this script is to find good prior parameters.
We define 'good' by measuring the iou between a sample of ground truth boxes from the training data.
The higher the IoU without any regression is, the better are the boxes.
"""
import numpy as np
import pandas as pd
import logging

import torch
from torchvision.ops import box_convert

from ecod.data.box2d.priors import prior_assignment_iou, PriorBox
from ecod.data.box2d.transforms import center_form_to_corner_form
from ecod.utils.general import ProgressLogger
from ecod.plot.priors import get_gt_seqs_and_boxes
from ecod.paths import random_move_debug_root, random_move_mnist36_root


class CumulativeDensityPoisson:
    def __init__(self, lam, n_max, n_samples=1000):
        self.rng = np.random.default_rng(511)
        vals = self.rng.poisson(lam=lam, size=n_samples)
        un, counts = np.unique(vals, return_counts=True)
        probs = np.zeros((n_max,), dtype=np.float64)
        probs[un] = counts
        probs /= n_samples
        self.probs = probs
        self.cum_density = np.cumsum(self.probs)

    def __getitem__(self, idx):
        return self.cum_density[idx]

    def __len__(self):
        return len(self.probs)


class BoundingBoxScore:
    def __init__(self, lam=1.0, n_max=1000, n_samples=10000):
        cum_dens_poisson = CumulativeDensityPoisson(lam=lam, n_max=n_max, n_samples=n_samples)
        self.scores = np.insert(cum_dens_poisson.cum_density.copy(), 0, 0)

    def __call__(self, idx):
        return self.scores[idx]


def mean_iou_over_priors(
    bboxs_col,
    shape_t,
    prior_feature_maps,
    prior_min_sizes,
    prior_max_sizes,
    prior_strides,
    prior_aspect_ratios,
    prior_clip=True,
    iou_func="giou",
    iou_thresh=0.5,
):
    score_func = BoundingBoxScore(lam=0.5)
    # define and assign priors
    priors = center_form_to_corner_form(
        PriorBox(
            shape_t=shape_t,
            prior_feature_maps=prior_feature_maps,
            prior_min_sizes=prior_min_sizes,
            prior_max_sizes=prior_max_sizes,
            prior_strides=prior_strides,
            prior_aspect_ratios=prior_aspect_ratios,
            prior_clip=True,
            debug_mode=False,
        )()
    )
    iou_means = []
    match_scores = []
    target_shape = shape_t[-2:]
    for idx in range(len(bboxs_col)):
        bboxs = torch.from_numpy(bboxs_col[idx].copy())
        # labels = torch.from_numpy(labels_col[idx].copy())
        bboxs[:, ::2] /= target_shape[1]
        bboxs[:, 1::2] /= target_shape[0]
        bboxs = box_convert(bboxs, "xywh", "xyxy")
        # n_priors x n_targets
        ious = prior_assignment_iou(bboxs, priors, iou_func)
        ious[ious < iou_thresh] = 0.0
        iou_means.append(ious.mean().numpy())
        n_matches_per_target = (ious >= iou_thresh).sum(0)
        score_matches = [score_func(ii) for ii in n_matches_per_target]
        scores_matches = np.mean(score_matches)
        match_scores.append(score_matches)

    return np.mean(iou_means), np.mean(match_scores)


def run_random_prior_search():
    results = {}
    idx = 0

    n_random_samples = 600
    max_aspect_ratios = 3
    max_aspect = 3.0
    aspect_distance_to_square = 0.2
    n_aspects_smaller_one = 1
    return_frames = True
    n_samples = 200
    root = random_move_mnist36_root
    bbox_suffix = "none"
    tvt = "train"

    shape_t = [4, 2, 360, 360]
    target_shape = shape_t[-2:]
    prior_feature_maps = [21, 9, 3]
    prior_clip = False
    iou_thresh = 0.5
    iou_func = "iou"
    prior_strides = [min(target_shape) / ff for ff in prior_feature_maps]

    bboxs_col = get_gt_seqs_and_boxes(root, bbox_suffix, tvt, target_shape, n_samples, return_frames)[1]
    rng = np.random.default_rng(52)
    random_min_sizes = rng.random(size=n_random_samples) * (0.2 - 0.01) + 0.01
    random_max_sizes = rng.random(size=n_random_samples) * (0.95 - 0.3) + 0.3
    random_n_priors = rng.integers(6, 8, size=n_random_samples)
    random_aspect_ratios = rng.random(size=n_random_samples * max_aspect_ratios).reshape(
        n_random_samples, max_aspect_ratios
    )
    random_aspect_ratios[:, :n_aspects_smaller_one] = (
        random_aspect_ratios[:, :n_aspects_smaller_one] * (1 - aspect_distance_to_square - 1.0 / max_aspect)
        + 1.0 / max_aspect
    )
    random_aspect_ratios[:, n_aspects_smaller_one:] = (
        random_aspect_ratios[:, :n_aspects_smaller_one] * (max_aspect - (1 + aspect_distance_to_square))
        + 1.0
        + aspect_distance_to_square
    )

    random_n_asp_ratios = rng.integers(1, max_aspect_ratios, size=n_random_samples)

    for ii in ProgressLogger(range(n_random_samples)):
        min_size = random_min_sizes[ii]
        max_size = random_max_sizes[ii]
        n_priors_per_layer = random_n_priors[ii]
        prior_aspect_ratios = [
            [1.0] + list(random_aspect_ratios[ii][: random_n_asp_ratios[ii]]) for _ in range(len(prior_feature_maps))
        ]

        prior_min_sizes = np.linspace(min_size, max_size, n_priors_per_layer)
        iou_mean, match_scores = mean_iou_over_priors(
            bboxs_col,
            shape_t,
            prior_feature_maps,
            prior_min_sizes,
            None,
            prior_strides,
            prior_aspect_ratios,
            prior_clip=prior_clip,
            iou_func=iou_func,
            iou_thresh=iou_thresh,
        )
        objective = (iou_mean + match_scores) / 2.0
        results[idx] = {
            "min_size": min_size,
            "max_size": max_size,
            "n_priors_pl": n_priors_per_layer,
            "aspect_ratios": prior_aspect_ratios,
            "objective": objective,
            "iou": iou_mean,
            "match_score": match_scores,
            "idx": idx,
        }
        idx += 1
    df = pd.DataFrame.from_dict(results, orient="index").sort_values("objective")
    return df


if __name__ == "__main__":
    logging.basicConfig()
    df = run_random_prior_search()
    df.to_csv("results.csv")
