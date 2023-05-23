"""Adapted partly from https://github.com/lufficc/SSD/blob/master/ssd/modeling/box_head/inference.py
MIT License Copyright (c) 2018 lufficc
"""
# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

from torchvision.ops.boxes import batched_nms

# PostProcessor(args_dict['shape_t'], args_dict['test_confidence_threshold'],
#                    args_dict['test_nms_threshold'], args_dict['test_max_per_image'])
# SimplePostProcessor(args_dict['shape_t'][-2:], args_dict['test_max_per_image'], args_dict['test_confidence_threshold'])


class SimplePostProcessor:
    def __init__(self, image_shape, max_per_image, confidence_threshold):
        self.max_per_image = max_per_image
        self.image_shape = image_shape
        self.confidence_threshold = confidence_threshold

    def __call__(self, detections):
        scores, boxes = detections
        test_confidence_threshold = self.confidence_threshold
        max_per_image = self.max_per_image
        height, width = self.image_shape
        batch_size = len(boxes)
        confidence, labels = torch.max(scores, 2)
        results = []
        for ii in range(batch_size):
            mask = labels[ii] > 0
            boxes_now = boxes[ii][mask]
            confidence_now = confidence[ii][mask]
            labels_now = labels[ii][mask]
            mask2 = confidence_now > test_confidence_threshold
            boxes_now = boxes_now[mask2]
            confidence_now = confidence_now[mask2]
            labels_now = labels_now[mask2]
            if len(confidence_now) > max_per_image:
                idxs_sort = torch.argsort(confidence_now)[:max_per_image]
                boxes_now = boxes_now[idxs_sort]
                confidence_now = confidence_now[idxs_sort]
                labels_now = labels_now[idxs_sort]
            boxes_now[:, 0::2] *= width
            boxes_now[:, 1::2] *= height
            res = dict(
                boxes=boxes_now,
                labels=labels_now,
                scores=confidence_now,
                shape=(height, width),
            )
        results.append(res)
        return results


class NMSPostProcessor:
    def __init__(self, shape_t, test_confidence_threshold, test_nms_threshold, test_max_per_image):
        super().__init__()
        self.width = shape_t[-1]
        self.height = shape_t[-2]
        self.test_confidence_threshold = test_confidence_threshold
        self.test_nms_threshold = test_nms_threshold
        self.test_max_per_image = test_max_per_image

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            # t_now = time.time()
            scores, boxes = (
                batches_scores[batch_id],
                batches_boxes[batch_id],
            )  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]
            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            indices = torch.nonzero((scores > self.test_confidence_threshold).int(), as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]
            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height
            # boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=0., max=self.width-1)
            # boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=0., max=self.height-1)
            # print("NMS prepare time: {:.3f}, n_boxes={}".format(time.time() - t_now, boxes.shape))
            # t_now = time.time()
            keep = batched_nms(boxes, scores, labels, self.test_nms_threshold)
            #  print("NMS time: {:.3f}, n_boxes={}".format(time.time() - t_now, boxes.shape))
            # keep only topk scoring predictions
            keep = keep[: self.test_max_per_image]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            res = dict(
                boxes=boxes,
                labels=labels,
                scores=scores,
                shape=(self.height, self.width),
            )
            results.append(res)
        return results


def get_post_processor(hparams):
    if hparams["postprocessor_name"] == "nms":
        return NMSPostProcessor(
            hparams["shape_t"],
            hparams["test_confidence_threshold"],
            hparams["test_nms_threshold"],
            hparams["test_max_per_image"],
        )
    elif hparams["postprocessor_name"] == "simple":
        return SimplePostProcessor(
            hparams["shape_t"][-2:],
            hparams["test_max_per_image"],
            hparams["test_confidence_threshold"],
        )
    else:
        raise ValueError(f"postprocessor_name has to be in ['nms', 'simple'] but is {hparams['postprocessor_name']}")
