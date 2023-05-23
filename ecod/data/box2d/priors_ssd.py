# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from itertools import product
import numpy as np

import torch


class PriorBoxSSD:
    def __init__(
        self,
        shape_t,
        prior_feature_maps,
        prior_min_sizes,
        prior_max_sizes,
        prior_strides,
        prior_aspect_ratios,
        prior_clip,
        debug_mode=False,
    ):
        self.height = shape_t[-2]
        self.width = shape_t[-1]
        self.feature_maps = prior_feature_maps
        self.min_sizes = prior_min_sizes
        self.max_sizes = prior_max_sizes
        self.strides = prior_strides
        self.aspect_ratios = prior_aspect_ratios
        self.clip = prior_clip
        self.debug_mode = debug_mode

    def __call__(self):
        """Generate SSD Prior Boxes.
        It returns the center, height and width of the priors. The values are relative to the image size
        Returns:
            priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                are relative to the image size.
            if debug_mode: priors list(num_priors_per_feature, 4): Can plot each feature map more easily
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            priors_f = self.get_prior_for_feature_map(f, k)
            if self.debug_mode:
                priors.append(priors_f)
            else:
                priors += priors_f
        if not self.debug_mode:
            priors = torch.tensor(priors, dtype=torch.float32)
            if self.clip:
                priors.clamp_(max=1, min=0)
        return priors

    def get_prior_for_feature_map(self, f, k):
        priors = []
        scale_x = self.width / self.strides[k]
        scale_y = self.height / self.strides[k]
        for i, j in product(range(f), repeat=2):
            # unit center x,y
            cx = (j + 0.5) / scale_x
            cy = (i + 0.5) / scale_y

            # small sized square box
            size = self.min_sizes[k]
            h = size / self.height
            w = size / self.width
            priors.append([cx, cy, w, h])

            # big sized square box
            size = np.sqrt(self.min_sizes[k] * self.max_sizes[k])
            h = size / self.height
            w = size / self.width
            priors.append([cx, cy, w, h])

            # change h/w ratio of the small sized box
            size = self.min_sizes[k]
            h = size / self.height
            w = size / self.width
            for ratio in self.aspect_ratios[k]:
                ratio = np.sqrt(ratio)
                priors.append([cx, cy, w * ratio, h / ratio])
                priors.append([cx, cy, w / ratio, h * ratio])
        if self.debug_mode:
            priors = torch.tensor(priors, dtype=torch.float32)
            if self.clip:
                priors.clamp_(max=1, min=0)
        return priors

    @classmethod
    def from_hparams(cls, hparams, debug_mode=False):
        return cls(
            hparams["shape_t"],
            hparams["prior_feature_maps"],
            hparams["prior_min_sizes"],
            hparams["prior_max_sizes"],
            hparams["prior_strides"],
            hparams["prior_aspect_ratios"],
            hparams["prior_clip"],
            debug_mode=debug_mode,
        )
