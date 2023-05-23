"""Adapted partly from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/anchor_generator.py
 (Apache License 2.0).
"""
import collections
from typing import List
import math

import torch
import torch.nn as nn
from torchvision.ops import box_convert


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.
    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.
    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(params, collections.abc.Sequence), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params


def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride,
        grid_width * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )
    shifts_y = torch.arange(
        offset * stride,
        grid_height * stride,
        step=stride,
        dtype=torch.float32,
        device=device,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class AnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim = 4
    """
    the dimension of each anchor box.
    """

    def __init__(self, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.
        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        self.num_features = len(self.strides)
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        cell_anchors = [self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)]
        return cell_anchors

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)
                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        buffers: List[torch.Tensor] = [x for x in self.cell_anchors]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size**2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, grid_sizes):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.
        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        # grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return anchors_over_all_feature_maps


def make_priors(grid_sizes, scales, aspect_ratios, strides, input_shape, debug_priors=False):
    anchors = []
    n_boxes_per_location = []
    if not hasattr(scales[0], "__len__"):
        scales = [scales for _ in range(len(grid_sizes))]
    for grid_size_layer, scales_layer, aspect_ratios_layer, strides_layer in zip(
        grid_sizes, scales, aspect_ratios, strides
    ):
        if not hasattr(grid_size_layer, "__len__"):
            grid_size_layer = [grid_size_layer, grid_size_layer]
        # sizes_layer = [ss * strides_layer for ss in scales_layer]
        sizes_layer = [ss * min(input_shape) for ss in scales_layer]
        anch_gen = AnchorGenerator(sizes_layer, aspect_ratios_layer, [strides_layer])
        anchors_layer = anch_gen.forward([grid_size_layer])[0]
        n_boxes_per_location.append(anch_gen.num_anchors[0])
        # normalize
        anchors_layer[:, ::2] /= input_shape[1]
        anchors_layer[:, 1::2] /= input_shape[0]
        # to center
        anchors_layer = box_convert(anchors_layer, in_fmt="xyxy", out_fmt="cxcywh")
        anchors.append(anchors_layer)

    if not debug_priors:
        anchors = torch.cat(anchors)
    return anchors, n_boxes_per_location


class PriorBoxRCNN:
    """Prior box generator for R-CNN-like prior boxes"""

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
        """Initialize

        Members:
            grid_sizes (List[List[Int]]): List of shapes of feature maps that are used for bbox heads
            scales (List[List[Float]]): Scales to use for prior boxes. Scales are normalized to input image, i.e.,
                0.5 is half the height and half the width of the full image size (regardless of feature map size).
            aspect_ratios (List[List[float]]): aspect ratios at each feature map. Don't forget to explicitly give '1.'
                                               to have a square feature map
            strides (List[Float]): strides of each feature map; usually, input_shape/feature_map_shape
            input_shape (List[Int]): list of length 2, giving (height, width)
            debug_priors (bool, optional): If True, return priors as list per feature map. Defaults to False.
        """
        self.grid_sizes = prior_feature_maps
        self.scales = prior_min_sizes
        self.aspect_ratios = prior_aspect_ratios
        self.strides = prior_strides
        self.input_shape = shape_t[-2:]
        self.clip = prior_clip
        self.debug_priors = debug_mode
        self.n_boxes_per_location = None

    @torch.no_grad()
    def __call__(self):
        priors, self.n_boxes_per_location = make_priors(
            self.grid_sizes,
            self.scales,
            self.aspect_ratios,
            self.strides,
            self.input_shape,
            self.debug_priors,
        )
        if self.clip:
            if self.debug_priors:
                new_priors = []
                for priors_per_map in priors:
                    new_priors.append(torch.clamp(priors_per_map, min=0, max=1.0))
                priors = new_priors
            else:
                priors = torch.clamp(priors, min=0.0, max=1.0)
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
