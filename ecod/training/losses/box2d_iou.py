# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from torchvision.ops import generalized_box_iou


def giou_torchvision(boxes_pred, boxes_gt, reduction="sum"):
    # boxes_gt = box_convert(boxes_gt, 'cxcywh', 'xyxy')
    giou = generalized_box_iou(boxes_pred, boxes_gt)
    loss = 0.5 * (1 - giou)
    if reduction == "sum":
        loss = loss.sum()
    return loss


def locations_center_to_xyxy(locations_center):
    x1 = locations_center[:, 0] - 0.5 * locations_center[:, 2]
    y1 = locations_center[:, 1] - 0.5 * locations_center[:, 3]
    x2 = locations_center[:, 0] + 0.5 * locations_center[:, 2]
    y2 = locations_center[:, 1] + 0.5 * locations_center[:, 3]
    return x1, x2, y1, y2


generalized_iou_loss = giou_torchvision
