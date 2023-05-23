"""Adapted partly from https://github.com/lufficc/SSD (MIT License)."""

import logging
import torch
import torch.nn.functional as F

from ecod.data.box2d.priors import convert_locations_to_boxes, PriorBox
from ecod.data.box2d.transforms import center_form_to_corner_form
from ecod.utils.data import get_dataset_attributes


class SSDBoxPredictor(torch.nn.Module):
    def __init__(self, boxes_per_location, out_channels, n_classes):
        super().__init__()
        self.boxes_per_location = boxes_per_location
        self.out_channels = out_channels
        self.n_classes = n_classes
        self.cls_headers = torch.nn.ModuleList()
        self.reg_headers = torch.nn.ModuleList()
        for level, (boxes_per_location, out_channels) in enumerate(zip(self.boxes_per_location, self.out_channels)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        return torch.nn.Conv2d(
            out_channels,
            boxes_per_location * self.n_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def reg_block(self, level, out_channels, boxes_per_location):
        return torch.nn.Conv2d(
            out_channels,
            boxes_per_location * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                # torch.nn.init.constant_(m.bias, 0.01)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.n_classes)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_pred




class SSDBoxHead(torch.nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict
        self.use_sigmoid_scores = not args_dict["no_box_head_sigmoid"]
        self.locations_to_boxes = self.args_dict["boxes_to_locations"]
        n_classes = get_dataset_attributes(args_dict["dataset"])["n_classes"]
        self.predictor = SSDBoxPredictor(
            args_dict["prior_boxes_per_location"],
            args_dict["prior_out_channels"],
            n_classes,
        )
        self.priors = None
        self.logger = logging.getLogger("SSDHEAD")
        self.logger.setLevel("INFO")

    def forward(self, features):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred):
        detections = (cls_logits, bbox_pred)
        return detections

    def _forward_test(self, cls_logits, bbox_pred):
        if self.locations_to_boxes:
            args_dict = self.args_dict
            if self.priors is None:
                self.priors = PriorBox(
                    args_dict["shape_t"],
                    args_dict["prior_feature_maps"],
                    args_dict["prior_min_sizes"],
                    args_dict["prior_max_sizes"],
                    args_dict["prior_strides"],
                    args_dict["prior_aspect_ratios"],
                    args_dict["prior_clip"],
                    debug_mode=False,
                )().to(bbox_pred.device)
                self.logger.info(f"Created head with {len(self.priors)} prior boxes")
            elif self.priors.device != bbox_pred.device:
                self.priors = self.priors.to(bbox_pred.device)
            boxes = convert_locations_to_boxes(
                bbox_pred,
                self.priors,
                self.args_dict["prior_center_variance"],
                self.args_dict["prior_size_variance"],
            )
            boxes = center_form_to_corner_form(boxes)
        else:
            boxes = bbox_pred
        if self.use_sigmoid_scores:
            scores = torch.sigmoid(cls_logits)
        else:
            scores = F.softmax(cls_logits, dim=2)
        detections = (scores, boxes)
        return detections
