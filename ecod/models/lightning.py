# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

""" Implements the RED model from 'Learning to Detect Objects with a 1 Megapixel Event Camera'.

"""

from pathlib import Path
import argparse
import pickle as pkl
from shutil import copy
import numpy as np
import logging

import pytorch_lightning as pl
from torchmetrics import Accuracy

import torch
from torch.nn.functional import cross_entropy, relu

from ecod.utils.general import optional
from ecod.training.callbacks import ImprovementTracker

from ecod.utils.data import get_dataset_attributes
from ecod.training.mlflow import log_mlflow_artifact
from ecod.utils.files import value_list_type
from ecod.models.heads.convlstm import ConvLSTMODHeads

from ecod.models.backbones.resnet import ResNetBackbone, ResNetBackboneSmall, ResNetBackboneBig, ResNetBackboneMed
from ecod.models.heads.seq2channels import Seq2ChannelsODHeads
from ecod.training.losses.box2d import MultiBoxLoss
from ecod.eval.evaluator import get_evaluator


def get_backbone(hparams, in_channels):
    # don't forget to add set_pretrained when adding new backbones
    name = hparams["bb_name"].lower()
    if name.startswith("resn"):
        if name == "resnet_small":
            return ResNetBackboneSmall(in_channels=in_channels)
        elif name == "resnet_big":
            return ResNetBackboneBig(in_channels=in_channels)
        elif name == "resnet_med":
            return ResNetBackboneMed(in_channels=in_channels)
        else:
            return ResNetBackbone(
                name,
                in_channels=in_channels,
                pool=False,
                load_from=hparams["load_from"],
            )
    else:
        raise ValueError(f"name has to be in ['res', 'dense', 'mobile' but is {name}")


def get_od_heads(hparams, output_shape):
    name = hparams["seq_od_name"]
    if name == "channels":
        return Seq2ChannelsODHeads(hparams, output_shape)
    elif name == "lstm":
        return ConvLSTMODHeads(hparams, output_shape)
    else:
        raise ValueError(f"seq_od_name has to be in 'channels', 'lstm' but is {name}")


class SeqODBackbone(torch.nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.steps_and_bins = (hparams["n_timesteps"], hparams["n_bins"])
        self.input_norm = torch.nn.BatchNorm2d(in_channels)
        self.bb = get_backbone(hparams, in_channels)

    def forward(self, inp):
        # inp: (batch, n_seqs, bins, channels, h, w)
        # process all seqs in parallel and merge channels with bins by reshaping
        bs = inp.shape[0]
        n_seqs = inp.shape[1]
        ins = inp.reshape(bs, n_seqs, -1, *inp.shape[-2:])
        outs = []
        for idx in range(ins.shape[1]):
            outs.append(self.bb(self.input_norm(ins[:, idx])))
        # outs is [(batch, bins*channels, h', w') * n_seqs], need (batch, n_seqs, bins*channels, h', w')
        out = torch.stack(outs, 1)
        return out


class LightningSeqOD(pl.LightningModule):
    def __init__(
        self,
        hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dataset_attrs = get_dataset_attributes(hparams["dataset"])

        in_channels = self.dataset_attrs["channels"] * hparams["n_bins"]
        self.backbone = SeqODBackbone(hparams, in_channels)
        self.backbone_channels = in_channels
        self.backbone_output_shape = self.get_backbone_output_shape()
        self.od_heads = get_od_heads(hparams, self.backbone_output_shape)
        self.save_hyperparameters(hparams)
        log = f"Backbone chans and feature maps: {hparams['prior_out_channels']}, {hparams['prior_feature_maps']}"
        self.cmd_logger = logging.getLogger("MODEL")
        self.cmd_logger.setLevel("INFO")
        self.cmd_logger.info(log)
        self.loss_evaluator = MultiBoxLoss(
            neg_pos_ratio=hparams["train_neg_pos_ratio"],
            bbox_loss_name=hparams["bbox_loss"],
            use_focal_loss=not hparams["no_focal_loss"],
        )
        self.loss_evaluator.use_hard_negative_mining = True
        # used during eval to keep state over batches
        self.persistent_state = None
        self.file_idx_last = -1
        self.bbox_evaluator = get_evaluator(hparams)
        self.improvement_tracker = ImprovementTracker("max")
        self.bboxs_path = None

    @torch.no_grad()
    def get_backbone_output_shape(self):
        shape = self.hparams["shape_t"]
        inp = torch.zeros((1, shape[0], self.hparams["n_bins"], *shape[1:]))
        # (batch, n_seqs, channels, h, w) -> (channels, h, w)
        return self.backbone(inp).shape[2:]

    def forward(self, inp, hx=None):
        out = self.backbone(inp)
        # (cls_logits, bbox_reg), hx with shape (batch, n_outs, n_priors, n_classes), (..., 4)
        # OR [[dict(boxes, labels, scores) for _ in range(n_seqs)] for _ in range(batches)] during eval
        out = self.od_heads(out, hx)
        return out

    def loss(self, logits, labels, bbox_reg_pred, bbox_reg_gt, step_name="train"):
        loss_reg, loss_class = self.loss_evaluator(logits, bbox_reg_pred, labels, bbox_reg_gt)
        loss_tot = loss_class + self.hparams["lambda_reg"] * loss_reg
        self.log(f"{step_name}_loss_class", loss_class, batch_size=1)
        self.log(f"{step_name}_loss_reg", loss_reg, batch_size=1)
        self.log(f"{step_name}_loss_tot", loss_tot, batch_size=1)
        return loss_tot

    def merge_batch_and_seq_channel(self, *args):
        out = []
        for arg in args:
            out.append(arg.reshape(arg.shape[0] * arg.shape[1], *arg.shape[2:]))
        return out

    def on_train_start(self) -> None:
        self.improvement_tracker.enable()

    def training_step(self, batch, batch_idx):
        (
            seqs,
            bboxs_gt,
            labels_gt,
            idxs_seq,
            idxs_frames,
            idxs_file,
            frame_times_ms,
        ) = batch
        (logits, bbox_reg_pred), _ = self.forward(seqs)
        n_seqs_pred = logits.shape[1]
        bboxs_gt = bboxs_gt[:, -n_seqs_pred:]
        labels_gt = labels_gt[:, -n_seqs_pred:]
        # merge batch and n_seqs channel to do loss eval for all sequence items in parallel
        logits, bbox_reg_pred, bboxs_gt, labels_gt = self.merge_batch_and_seq_channel(
            logits, bbox_reg_pred, bboxs_gt, labels_gt
        )
        loss = self.loss(logits, labels_gt, bbox_reg_pred, bboxs_gt)
        outputs = {
            "loss": loss,
        }
        DEBUG_OUTS = True
        if batch_idx == 0 and DEBUG_OUTS:
            pkl_path = Path(self.hparams["temp_dir"]) / f"training_outs_{batch_idx}.pkl"
            data = {
                "seqs": seqs,
                "bboxs_gt": bboxs_gt,
                "labels_gt": labels_gt,
                "idxs_seq": idxs_seq,
                "idxs_frames": idxs_frames,
                "logits": logits,
                "bboxs_pred": bbox_reg_pred,
                "loss": loss,
            }
            data = {key: val.cpu().detach().numpy() for key, val in data.items()}
            with open(pkl_path, "wb") as hd:
                pkl.dump(data, hd)
        return outputs

    def init_persistent_state(self, batch_size, device):
        hx = self.od_heads.get_hidden_state(batch_size, device)
        return hx

    def get_persistent_state_batch_size(self):
        if self.persistent_state is None:
            return -1
        if self.hparams["seq_od_name"] == "lstm":
            # list of length of ConvLSTM, each entry is tuple (hidden, cell) state
            return self.persistent_state[0][0].shape[0]
        else:
            return self.persistent_state.shape[0]

    @torch.no_grad()
    def detach_persistent_state(self, persistent_state):
        if self.hparams["seq_od_name"] == "lstm":
            return [(hh.detach().clone(), cc.detach().clone()) for hh, cc in persistent_state]
        else:
            return persistent_state.detach().clone()

    def check_persistent_state(self):
        if self.hparams["seq_od_name"] == "lstm":
            for state in self.persistent_state:
                for ss in state:
                    if torch.isnan(ss).any():
                        raise RuntimeError("Persistent state contains nans")
                    if torch.isinf(ss).any():
                        raise RuntimeError("Persistent state contains infs")
        else:
            if torch.isnan(self.persistent_state).any():
                raise RuntimeError("Persistent state contains nans")
            if torch.isinf(self.persistent_state).any():
                raise RuntimeError("Persistent state contains infs")

    def val_test_step(self, batch, batch_idx, name):
        seqs, bboxs_gt, labels_gt, idxs, sample_idxs, file_idxs, frame_times_ms = batch
        if len(seqs) != 1:
            raise RuntimeError("Right now, batch_size > 1 is not implemented for validation/test")
        bs_pers = self.get_persistent_state_batch_size()
        if batch_idx == 0 or seqs.shape[0] != bs_pers or self.file_idx_last != file_idxs[0]:
            self.persistent_state = self.init_persistent_state(seqs.shape[0], seqs.device)
            self.file_idx_last = file_idxs[0]
        (logits, bbox_reg_pred), persistent_state = self.forward(seqs, self.persistent_state)
        self.persistent_state = self.detach_persistent_state(persistent_state=persistent_state)
        self.check_persistent_state()
        self.bbox_evaluator(
            logits,
            bbox_reg_pred,
            seqs,
            bboxs_gt,
            labels_gt,
            idxs,
            sample_idxs,
            file_idxs,
            frame_times_ms,
        )
        return None

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, "test")

    def val_test_end(self, step_outputs, name):
        self.bboxs_path, metrics = self.bbox_evaluator.save_and_evaluate(name)
        self.bbox_evaluator.reset()
        self.log_dict(metrics, batch_size=1)
        # log val_bboxs.h5 to mlflow if mAP improved
        if name == "val":
            self.improvement_tracker(metrics["val_mAP"])
            if self.improvement_tracker.improved is not None:
                if self.improvement_tracker.improved:
                    self.cmd_logger.info(f"Copying {self.bboxs_path.name} because val_mAP improved")
                    best_val_bboxs_path = self.bboxs_path.parent / f"best_{self.bboxs_path.name}"
                    copy(self.bboxs_path, best_val_bboxs_path)
                    log_mlflow_artifact(self.logger, best_val_bboxs_path)
                else:
                    self.cmd_logger.info("Not copying val bboxs")
            else:
                self.cmd_logger.warn(
                    "ImprovementTracker.improved is None; Can't determine if state should be updated. "
                    "Please call the tracker beforehand."
                )
        else:
            log_mlflow_artifact(self.logger, self.bboxs_path)
        self.persistent_state = None
        self.file_idx_last = -1
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        self.val_test_end(validation_step_outputs, "val")
        return None

    def test_epoch_end(self, test_step_outputs):
        self.val_test_end(test_step_outputs, "test")
        return None

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        milestones = (np.array([0.1, 0.6, 0.85, 0.9]) * self.hparams["max_epochs"]).astype(int)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3, last_epoch=-1)
        return [opt], [scheduler]

    def training_epoch_end(self, outputs) -> None:
        # log learning rate
        if not self.hparams["test"]:
            lr = [pp["lr"] for pp in self.optimizers().param_groups]
            if len(lr) > 1:
                self.log_dict(
                    {f"learning_rate_{ii}": lr[ii] for ii in range(len(lr))},
                    batch_size=1,
                )
            else:
                self.log("learning_rate", lr[0], batch_size=1)
        # switch between random and hard negative every epoch
        # self.loss_evaluator.use_hard_negative_mining = not self.loss_evaluator.use_hard_negative_mining
        return None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = add_general_args(parent_parser)
        # ssd args
        parser.add_argument("--no_box_head_sigmoid", action="store_true")
        parser.add_argument("--test_confidence_threshold", type=float, default=0.01)
        parser.add_argument("--test_nms_threshold", type=float, default=0.45)
        parser.add_argument("--test_max_per_image", type=int, default=50)
        parser.add_argument("--test_not_use_07_metric", action="store_true")
        parser.add_argument("--train_neg_pos_ratio", type=float, default=3)
        parser.add_argument("--train_iou_threshold", type=float, default=0.5)
        parser.add_argument("--train_iou_func", type=str, default="iou", choices=["iou", "giou", "igt"])
        parser.add_argument("--no_focal_loss", action="store_true")
        # add FPN between features_list and ssd_head; Somehow makes results much worse
        parser.add_argument("--ssd_use_fpn", action="store_true")
        # scaling factor for regression loss (relative to classification loss)
        parser.add_argument("--lambda_reg", type=float, default=1.0)
        # seq net args
        parser.add_argument("--seq_od_name", type=str, default="channels", choices=["channels", "lstm"])
        parser.add_argument("--seq_n_buffers", type=int, default=2)
        parser.add_argument("--hidden_dims", type=int, default=[256, 256], nargs="+")
        # scaling params for bbox->locations transform
        parser.add_argument("--prior_center_variance", type=float, default=0.1)
        parser.add_argument("--prior_size_variance", type=float, default=0.2)
        # defines prior boxes; out_channels, feature_maps, boxes_per_location, strides are set automatically if not
        #  provided
        parser.add_argument("--prior_min_sizes", type=float, nargs="+", default=None)
        parser.add_argument("--prior_max_sizes", type=float, nargs="+", default=None)
        parser.add_argument("--prior_aspect_ratios", type=value_list_type, default=[[0.5, 1.0, 3.0]])
        parser.add_argument("--prior_not_clip", action="store_true")
        # these are set automatically if not provided (derived from the model)
        parser.add_argument("--prior_out_channels", type=int, nargs="+", default=None)
        parser.add_argument("--prior_feature_maps", type=int, nargs="+", default=None)
        parser.add_argument("--prior_boxes_per_location", type=int, nargs="+", default=None)
        parser.add_argument("--prior_strides", type=int, nargs="+", default=None)
        parser.add_argument(
            "--prior_assignment_debug_mode",
            action="store_true",
            help="Return prior boxes instead of gt boxes for regression; This only makes sense for debugging",
        )
        # these prior args only work with R-CNN prior; They override some of the SSD prior args if they are not None
        # overwrites prior_min_sizes
        parser.add_argument("--prior_scales", type=value_list_type, default=[0.05,0.11,0.17,0.28,0.42,0.56])
        # backbone args
        parser.add_argument("--bb_name", type=str, default="resnet18")

        # training args
        # number of steps to take into account during training; will rollout for n_timesteps steps, but only take last
        #  train_n_predictions steps outputs into account
        parser.add_argument("--train_n_predictions", type=int, default=1)
        parser.add_argument("--bbox_loss", type=str, default="smooth_l1", choices=["giou", "smooth_l1"])
        parser.add_argument(
            "--boxes_to_locations",
            action="store_true",
            help="DONT USE MANUALLY, IS SET AUTOMATICALLY",
        )
        # augmentation args
        parser.add_argument("--random_crop", action="store_true")
        parser.add_argument("--random_mirror", action="store_true")
        parser.add_argument("--simple_post_process", action="store_true")
        # bbox evaluator args
        parser.add_argument("--evaluator_name", default="standard", choices=["standard"])
        parser.add_argument("--postprocessor_name", default="nms", choices=["nms", "simple"])
        return parser


def add_general_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--shape_t", type=int, nargs=4, default=[4, 2, 360, 360])
    parser.add_argument("--n_timesteps", default=4, type=int)
    parser.add_argument("--n_channels", default=2, type=int)
    parser.add_argument("--width", default=360, type=int)
    parser.add_argument("--height", default=360, type=int)
    parser.add_argument("--name", type=str, default="delme")
    help_load = "Continue training of a given checkpoint (includes loading optimizer, dataset, configuration)."
    parser.add_argument("--load_from", type=str, default=None, help=help_load)
    parser.add_argument("--weights_path", type=str, default=None, help="Load weights from a given checkpoint.")
    parser.add_argument("--temp_dir", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    # training args
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    return parser
