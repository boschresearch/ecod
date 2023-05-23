# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torchvision.ops import box_area


def normalized_boxes_to_absolute(boxes, shape, copy=True):
    """

    Args:
        boxes: xyxy boxes
        shape: [height, width]
        copy:

    Returns:

    """
    if copy:
        boxes = boxes.copy()
    boxes[:, 0::2] *= shape[1]
    boxes[:, 1::2] *= shape[0]
    return boxes


def normalize_boxes(boxes, shape, copy=True):
    if copy:
        if hasattr(boxes, "copy"):
            boxes = boxes.copy()
        else:
            boxes = boxes.clone()
    boxes[:, 0::2] /= shape[1]
    boxes[:, 1::2] /= shape[0]
    return boxes


def concatenate_list_of_dicts(list_of_dicts):
    concat_dict = {key: [] for key in list_of_dicts[0].keys()}
    for ii in range(len(list_of_dicts)):
        # always index 0 because list shrinks with every `pop`
        output = list_of_dicts.pop(0)
        for key in concat_dict:
            concat_dict[key] += output[key]
    return concat_dict


def center_form_to_corner_form(locations):
    """
    corner form: tlxy brxy
    center form: cxy wh
    """
    return torch.cat(
        [
            locations[..., :2] - locations[..., 2:] / 2,
            locations[..., :2] + locations[..., 2:] / 2,
        ],
        locations.dim() - 1,
    )


def corner_form_to_center_form(boxes):
    """
    corner form: tlxy brxy
    center form: cxy wh
    """
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]],
        boxes.dim() - 1,
    )


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def intersection_over_gt(boxes_gt: torch.Tensor, boxes_prior: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-ground truth of boxes.
    In contrast to IoU, the area outside the bounding box but inside the prior box is not punished.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes_gt (Tensor[N, 4])
        boxes_prior (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoG values for every element in boxes_gt
                            and boxes_prior
    """
    area1 = box_area(boxes_gt)

    lt = torch.max(boxes_gt[:, None, :2], boxes_prior[:, :2])  # [N,M,2]
    rb = torch.min(boxes_gt[:, None, 2:], boxes_prior[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / area1[:, None]
    return iou
