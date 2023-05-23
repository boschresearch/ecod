# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from ecod.training.losses.box2d_iou import generalized_iou_loss


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def random_negatives(labels, neg_pos_ratio):
    """
    Selects random negatives and all positives on a batch level. Faster than hard_negative_mining above.
    """
    pos_mask = labels.flatten() > 0
    num_pos = pos_mask.long().sum()
    num_neg = int(num_pos * neg_pos_ratio)
    idxs_neg = torch.arange(len(pos_mask))[~pos_mask]
    idxs_select = torch.randperm(len(idxs_neg))[:num_neg]
    pos_mask[idxs_neg[idxs_select]] = True
    return pos_mask


class MultiBoxLoss(torch.nn.Module):
    def __init__(self, neg_pos_ratio, bbox_loss_name, use_focal_loss=False):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.use_hard_negative_mining = True
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss and not self.use_hard_negative_mining:
            raise ValueError("use_focal_loss can't be true when use_hard_negative_mining is False")
        self.bbox_loss_name = bbox_loss_name

    def classification_loss_hard_negative(self, confidence, labels):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # confidence has shape (b, n_priors, n_classes)
            # selecting log_softmax[:, :, 0] gives loss for background class,
            # which is all we need to select the "most negative" priors
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)
        if self.use_focal_loss:
            labels = F.one_hot(labels[mask], num_classes).to(torch.float32)
            confidence = confidence[mask, :]
            classification_loss = sigmoid_focal_loss(confidence, labels, reduction="sum")
        else:
            confidence = confidence[mask, :]
            classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction="sum")
        return classification_loss

    def classification_loss_random_negatives(self, confidence, labels):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            mask = random_negatives(labels, self.neg_pos_ratio)

        confidence = confidence.reshape(-1, num_classes)[mask]
        labels = labels.flatten()[mask]
        if self.use_focal_loss:
            labels = F.one_hot(labels, num_classes).to(torch.float32)
            classification_loss = sigmoid_focal_loss(confidence, labels, reduction="sum")
        else:
            classification_loss = F.cross_entropy(confidence, labels, reduction="sum")
        return classification_loss

    def bbox_loss_smooth_l1(self, labels, predicted_locations, gt_locations):
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction="sum")
        num_pos = len(gt_locations)
        return smooth_l1_loss / num_pos, num_pos

    def bbox_loss_giou(self, labels, predicted_locations, gt_locations):
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        # predicted_locations = torch.sigmoid(predicted_locations)
        # mask = (predicted_locations[:, :2] <= predicted_locations[:, 2:]).all(1)
        # predicted_locations = predicted_locations[mask]
        # gt_locations = gt_locations[mask]
        return generalized_iou_loss(predicted_locations, gt_locations, reduction="mean"), len(gt_locations)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        if self.use_hard_negative_mining:
            classification_loss = self.classification_loss_hard_negative(confidence, labels)
        else:
            classification_loss = self.classification_loss_random_negatives(confidence, labels)
        if self.bbox_loss_name == "smooth_l1":
            bbox_loss, num_pos = self.bbox_loss_smooth_l1(labels, predicted_locations, gt_locations)
        elif self.bbox_loss_name == "giou":
            bbox_loss, num_pos = self.bbox_loss_giou(labels, predicted_locations, gt_locations)
        else:
            raise ValueError("bbox_loss_name has to be 'smooth_l1', 'giou'")
        return bbox_loss, classification_loss / num_pos
