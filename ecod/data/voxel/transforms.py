# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import random


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # cutting out around one bbox
            "focus",
        )

    def __call__(self, seq, boxes, labels):
        """

        Args:
            seq ([type]): [description]
            boxes ([type]): Shape xywh
            labels ([type]): [description]

        Returns:
            [type]: [description]
        """
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return seq, boxes, labels
        height, width = seq.shape[-2:]
        # randomly choose a mode
        mode = random.choice(self.sample_options)
        if mode is None:
            return seq, boxes, labels
        boxes = boxes.copy()
        # xywh to xyxy
        boxes[:, -2:] = boxes[:, 0:2] + boxes[:, -2:]
        idx = np.random.choice(len(boxes))
        focus_bbox = boxes[idx].copy()
        xy = np.random.uniform([0, 0], focus_bbox[:2], size=2)
        target_wh = np.array(seq.shape[-2:])[::-1]
        shape_max = target_wh - 1 - xy
        wh_min = np.max([focus_bbox[-2:] - xy, np.array([0.2 * width, 0.2 * height])], 0)
        if (wh_min >= shape_max).any():
            # xyxy to xywh
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
            return seq, boxes, labels
        wh = np.random.uniform(wh_min, shape_max, size=2)
        rect = np.concatenate([xy, xy + wh], 0).astype(int)
        # cut the crop from the image
        current_image = seq[:, :, :, rect[1] : rect[3], rect[0] : rect[2]]
        # keep overlap with gt box IF center in sampled patch
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
        # mask in that both m1 and m2 are true
        mask = m1 * m2
        # take only matching gt boxes
        current_boxes = boxes[mask, :].copy()
        # take only matching gt labels
        current_labels = labels[mask]
        # should we use the box left and top corner or the crop's
        current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, :2] -= rect[:2]
        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, 2:] -= rect[:2]
        # xyxy to xywh
        current_boxes[:, 2:] = current_boxes[:, 2:] - current_boxes[:, :2]
        return current_image, current_boxes, current_labels


class RandomHorizontalMirror(object):
    def __call__(self, seq, boxes, classes):
        width = seq.shape[-1]
        if random.randint(0, 2) == 1:
            seq = seq[..., ::-1]
            boxes = boxes.copy()
            br_xs = width - 1 - boxes[:, 0]
            tl_xs = br_xs - boxes[:, 2]
            boxes[:, 0] = tl_xs
        return seq, boxes, classes
