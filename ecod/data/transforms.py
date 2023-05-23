"""
Adapted partly from https://github.com/amdegroot/ssd.pytorch (MIT License) and partly from
 https://github.com/lufficc/SSD (MIT License).
"""

import warnings
import numpy as np
import torch
from torchvision.ops import box_convert

from ecod.data.box2d.priors import (
    assign_priors,
    convert_boxes_to_locations,
    convert_locations_to_boxes,
)
from ecod.data.box2d.transforms import corner_form_to_center_form, center_form_to_corner_form
from ecod.events.voxel import VoxelGridResizerAvgPool


class SSDTargetTransform:
    """Transform gt_boxes (N, 4) to (n_priors, 4)
    This HAS TO BE done *after* rescaling image and boxes, such that target_image_size normalizes correctly
    """

    def __init__(
        self,
        center_form_priors,
        target_image_size,
        center_variance,
        size_variance,
        iou_threshold,
        only_best_priors=False,
        debug_mode_priors=False,
        boxes_to_locations=True,
        iou_func="giou",
    ):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = center_form_to_corner_form(center_form_priors)
        self.target_image_size = target_image_size
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.only_best_priors = only_best_priors
        self.debug_mode_priors = debug_mode_priors
        self.boxes_to_locations = boxes_to_locations
        self.iou_func = iou_func

    def __call__(self, gt_boxes, gt_labels):
        """
        gt_boxes: tlxywh, normalized to 1
        gt_labels: range(1, n_classes+1), because label 0 will be __background__
        """
        # len = 0 can happen for sequences, because could be that don't have any bboxs at one point of the seq
        if len(gt_boxes) == 0:
            locations = torch.zeros_like(self.corner_form_priors)
            labels = torch.zeros(
                (len(self.corner_form_priors),),
                device=self.corner_form_priors.device,
                dtype=torch.long,
            )
        else:
            if type(gt_boxes) is np.ndarray:
                gt_boxes = torch.from_numpy(gt_boxes)
            if type(gt_labels) is np.ndarray:
                gt_labels = torch.from_numpy(gt_labels)
            if (gt_boxes.max(1)[0] < 1.0).any():
                raise RuntimeError("ground truth boxes are already rescaled between [0, 1), this should happen here!")
            # divide by *target_image_size* and not *original_image_size*, because we already rescaled at this step
            gt_boxes[:, ::2] /= self.target_image_size[1]
            gt_boxes[:, 1::2] /= self.target_image_size[0]
            gt_boxes = box_convert(gt_boxes, "xywh", "xyxy")
            boxes, labels = assign_priors(
                gt_boxes,
                gt_labels,
                self.corner_form_priors,
                self.iou_threshold,
                self.only_best_priors,
                self.debug_mode_priors,
                self.iou_func,
            )
            if self.boxes_to_locations:
                boxes = corner_form_to_center_form(boxes)
                locations = convert_boxes_to_locations(
                    boxes,
                    self.center_form_priors,
                    self.center_variance,
                    self.size_variance,
                )
            else:
                locations = boxes
        return locations, labels


class SSDTargetBackTransform(SSDTargetTransform):
    def __call__(self, locations, gt_labels, target_image_size=None):
        if type(locations) is np.ndarray:
            locations = torch.from_numpy(locations)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        if self.boxes_to_locations:
            boxes = convert_locations_to_boxes(
                locations,
                self.center_form_priors,
                self.center_variance,
                self.size_variance,
            )
            boxes = center_form_to_corner_form(boxes)
        else:
            boxes = locations
        if target_image_size is not None:
            boxes[:, ::2] *= target_image_size[1]
            boxes[:, 1::2] *= target_image_size[0]
        else:
            boxes[:, ::2] *= self.target_image_size[1]
            boxes[:, 1::2] *= self.target_image_size[0]
            warnings.warn(
                "New feature: Boxes are scaled back to image size in backtransform. " "Be careful if this is desired"
            )
        return boxes, gt_labels


class SSDTargetTransformWithoutPriors:
    def __init__(self, max_boxes=10000):
        self.max_boxes = max_boxes

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        n_missing = self.max_boxes - len(gt_boxes)
        if n_missing < 0:
            raise ValueError("Cannot transform bounding boxes")
        boxes_add = torch.full(
            (n_missing, gt_boxes.shape[-1]),
            0,
            dtype=gt_boxes.dtype,
            device=gt_boxes.device,
        )
        boxes = torch.cat([gt_boxes, boxes_add], 0)
        labels_add = torch.full((n_missing,), 0, dtype=gt_labels.dtype, device=gt_labels.device)
        labels = torch.cat([gt_labels, labels_add], 0)

        return boxes, labels


class VoxelGridTransform:
    def __init__(self, target_shape, norm_factor=9.52):
        """

        Args:
            target_shape (tuple): Shape to resize to
            norm_factor (float, optional): Normalizing factor applied to all sequence items of the voxel grid.
                Defaults to 9.52, which is the 95th percentile of all nonzero elements of the two seconds, one object
                random move mnist dset.
        """
        self.target_shape = target_shape
        self.norm_factor = norm_factor
        self.resizer = VoxelGridResizerAvgPool(target_shape)

    def bad_sk_resizer(self, seq):
        """
        Resizing like this takes 270 ms for a sequence of size (5, 4, 244, 244), while it takes 3 ms with pytorch
        """
        from skimage.transform import resize

        org_shape = seq.shape
        gg = seq.reshape(-1, *seq.shape[-2:])
        resized = resize(gg, [gg.shape[0], *self.target_shape])
        return resized.reshape(*org_shape[:-2], *resized.shape[-2:])

    def __call__(self, seq, bboxs, labels):
        orig_shape = seq.shape[-2:]
        target_shape = self.target_shape
        if orig_shape[0] != target_shape[0] or orig_shape[1] != target_shape[1]:
            seq = self.resizer(seq)
            seq /= self.norm_factor
            bboxs[:, ::2] *= target_shape[1] / orig_shape[1]
            bboxs[:, 1::2] *= target_shape[0] / orig_shape[0]
        return torch.from_numpy(seq).float(), bboxs, labels
