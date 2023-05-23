"""Adapted partly from https://github.com/lufficc/SSD (MIT License)."""

import math
import warnings

import torch
from torchvision.ops import generalized_box_iou

from ecod.data.box2d.priors_rcnn import PriorBoxRCNN
from ecod.data.box2d.priors_ssd import PriorBoxSSD
from ecod.data.box2d.transforms import intersection_over_gt, iou_of

# use to switch between SSD-style and R-CNN-style bounding box definition
# PriorBox = PriorBoxSSD
PriorBox = PriorBoxRCNN


def check_devices(locations, priors):
    if locations.device != priors.device:
        priors = priors.to(locations.device)
    return priors, locations


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    # FIXME: I think the height and width should be swapped, because it is [c_x, c_y] and [h, w] but w corresponds to x
    #  But does not matter because everything is symmetric right now anyway
    return torch.cat(
        [
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:],
        ],
        dim=locations.dim() - 1,
    )


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat(
        [
            (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
            torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance,
        ],
        dim=center_form_boxes.dim() - 1,
    )


def prior_assignment_iou(gt_boxes, corner_form_priors, iou_func):
    if iou_func == "giou":
        ious = generalized_box_iou(corner_form_priors, gt_boxes)
    elif iou_func == "iou":
        ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    elif iou_func == "igt":
        ious = intersection_over_gt(gt_boxes, corner_form_priors).T
    else:
        raise ValueError(f"iou_func has to be in ['iou', 'giou', 'igt'], but is {iou_func}")
    return ious


def assign_priors(
    gt_boxes,
    gt_labels,
    corner_form_priors,
    iou_threshold,
    only_best_priors=False,
    debug_mode=False,
    iou_func="giou",
):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = prior_assignment_iou(gt_boxes, corner_form_priors, iou_func=iou_func)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)
    if any(best_prior_per_target <= 0) and len(gt_boxes) > 0:
        warnings.warn(
            "Your prior boxes are defined in a way that there are gt boxes that do not overlap "
            "at all with any prior box"
        )
    if only_best_priors:
        boxes = torch.zeros((len(corner_form_priors), 4))
        if debug_mode:
            boxes[best_prior_per_target_index] = corner_form_priors[best_prior_per_target_index]
        else:
            boxes[best_prior_per_target_index] = gt_boxes
        labels = torch.zeros((len(corner_form_priors),), dtype=gt_labels.dtype)
        labels[best_prior_per_target_index] = gt_labels
    else:
        # size: num_priors
        best_target_per_prior, best_target_per_prior_index = ious.max(1)

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
        # size: num_priors
        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
        if debug_mode:
            boxes = corner_form_priors
        else:
            boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def calculate_number_of_boxes_per_feature_map(prior_aspect_ratios, prior_scales=None):
    if prior_scales is None:
        boxes_per_location = [2 + 2 * len(pp) for pp in prior_aspect_ratios]
    else:
        if hasattr(prior_scales[0], "__len__"):
            boxes_per_location = [len(pp) * len(ss) for ss, pp in zip(prior_scales, prior_aspect_ratios)]
        else:
            boxes_per_location = [len(pp) * len(prior_scales) for pp in prior_aspect_ratios]
    return boxes_per_location


def patch_prior_params_from_conv_net(args_dict, out_channels, feature_maps):
    boxes_per_location = calculate_number_of_boxes_per_feature_map(
        args_dict["prior_aspect_ratios"], args_dict["prior_scales"]
    )
    strides = [args_dict["shape_t"][-1] / ff for ff in feature_maps]
    args_dict["prior_out_channels"] = out_channels
    args_dict["prior_feature_maps"] = feature_maps
    args_dict["prior_boxes_per_location"] = boxes_per_location
    args_dict["prior_strides"] = strides
    return args_dict


def check_prior_params(args_dict):
    str_to_check = [
        "prior_out_channels",
        "prior_feature_maps",
        "prior_boxes_per_location",
        "prior_aspect_ratios",
        "prior_strides",
    ]
    extra_check = ["prior_min_sizes", "prior_max_sizes"]
    lens = [len(args_dict[ss]) for ss in str_to_check]
    if args_dict["prior_scales"] is None:
        for key in extra_check:
            lens.append(len(args_dict[key]))
    else:
        for ss in args_dict["prior_scales"]:
            if hasattr(ss, "__len__"):
                failed = any([kk > 0.999 or kk < 0 for kk in ss])
            else:
                failed = ss > 0.999 or ss < 0
            if failed:
                raise ValueError(f"Prior scales have to be in range [0, 0.999), but are {args_dict['prior_scales']}")
    if any([lens[0] != ll for ll in lens]):
        prior_params = {kk: args_dict[kk] for kk in str_to_check + extra_check}
        raise ValueError(f"All lengths of prior parameters have to be the same, but are: {prior_params}")
