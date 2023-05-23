# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import logging
import argparse
import pickle as pkl
import numpy as np
import matplotlib
import mlflow

import torch
from torchvision.ops import box_iou

from ecod.utils.files import makedirs, load_args_dict
from ecod.eval.evaluator import (
    load_and_postprocess,
    save_predicted_boxes_stacked,
    load_predicted_boxes_stacked,
    get_evaluator,
)
from ecod.training.mlflow import set_mlflow_tracking, mlflow_create_or_get_experiment_id
from ecod.utils.preparation import prepare_logging



def manual_iou_calculation(bbox_evaluator):
    for ii in range(len(bbox_evaluator.step_outputs["labels_gt"])):
        bboxs_gt_now = bbox_evaluator.step_outputs["bboxs_gt"][ii]
        bboxs_pred_now = bbox_evaluator.step_outputs["bboxs_pred"][ii]
        labels_gt_now = bbox_evaluator.step_outputs["labels_gt"][ii]
        labels_pred_now = bbox_evaluator.step_outputs["labels_pred"][ii]
        iou = box_iou(torch.from_numpy(bboxs_gt_now), torch.from_numpy(bboxs_pred_now)).numpy()
        top_k = 5
        top_k_iou_idxs = np.argsort(-iou, 1)[:, :top_k]
        for jj in range(len(bboxs_gt_now)):
            top_idxs = top_k_iou_idxs[jj]
            print(
                "gt label, predicted labels with highest iou, iou",
                labels_gt_now[jj],
                labels_pred_now[top_idxs],
                iou[jj, top_idxs],
            )


def get_random_file_name(length=5):
    letter_strings = np.concatenate([np.arange(65, 91, 1), np.arange(100, 123, 1)])
    file_name = "".join([chr(ss) for ss in np.random.choice(letter_strings, size=length)])
    return file_name


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--savedir", required=True)
    parser.add_argument("--val_test", default="val", choices=["val", "test", "both"])
    parser.add_argument("--load_postprocessed", action="store_true")
    parser.add_argument("--process_only_n_percent", default=None, type=float)
    # overwrite args
    parser.add_argument("--postprocessor_name", default=None)
    parser.add_argument("--test_confidence_threshold", default=None)
    parser.add_argument("--test_nms_threshold", default=None)
    parser.add_argument("--test_max_per_image", default=None)
    parser.add_argument("--box_mem", action="store_true")
    parser.add_argument("--conf_thresh", type=float, default=0.001)
    parser.add_argument("--forget_thresh", type=float, default=0.001)
    parser.add_argument("--delete_thresh", type=float, default=100)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--load_counts", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--idx_start", type=int, default=0)
    # mlflow args
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args(args=args)
    return args


def main():
    matplotlib.use("agg")
    prepare_logging()
    set_mlflow_tracking()

    logger = logging.getLogger("OFFEVAL")
    logger.setLevel("INFO")
    args = parse_args()
    exp_dir = Path(args.expdir)
    savedir = Path(args.savedir)
    box_mem_savepath = savedir
    box_mem_load_counts = args.load_counts
    makedirs(savedir)
    try:
        args_dict = load_args_dict(exp_dir)
    except RuntimeError as e:
        logger.warning(f"Could not find best model because '{e}'. Trying to load args_dict from pickle.")
        with open(exp_dir / "args_dict.pkl", "rb") as hd:
            args_dict = pkl.load(hd)
    if args.name is not None:
        exp_id = mlflow_create_or_get_experiment_id(
            args.name, args_dict["exp"], args_dict["task"], args_dict["dataset"]
        )
        mlflow.start_run(experiment_id=exp_id)
        for key, value in args_dict.items():
            mlflow.log_param(key, value)
        for key, value in vars(args).items():
            key = f"off_eval_{key}"
            mlflow.log_param(key, value)

    shape = args_dict["shape_t"][-2:]
    val_test_name = args.val_test
    if val_test_name in ["val", "test"]:
        val_test_name = [val_test_name]
    elif val_test_name == "both":
        val_test_name = ["val", "test"]
    else:
        raise ValueError(f"val_test cannot be {val_test_name}")

    for val_test in val_test_name:

        savepath = savedir / f"{args.val_test}_bboxs_postproc.h5"
        if args.load_postprocessed:
            bbox_evaluator = get_evaluator(args_dict)
            bbox_evaluator.step_outputs = load_predicted_boxes_stacked(savepath)
            bbox_evaluator.post_processed = True
        else:
            bbox_evaluator, args_dict = load_and_postprocess(
                exp_dir,
                val_test,
                args_dict,
                postprocessor_name=args.postprocessor_name,
                test_confidence_threshold=args.test_confidence_threshold,
                test_nms_threshold=args.test_nms_threshold,
                test_max_per_image=args.test_max_per_image,
                process_only_n_percent=args.process_only_n_percent,
                box_mem=args.box_mem,
                box_mem_conf_thresh=args.conf_thresh,
                box_mem_forget_thresh=args.forget_thresh,
                box_mem_delete_thresh=args.delete_thresh,
                box_mem_iou_thresh=args.iou_thresh,
                box_mem_savepath=box_mem_savepath,
                box_mem_load_counts=box_mem_load_counts,
                box_mem_plot=args.plot,
                idx_start=args.idx_start,
            )
            save_predicted_boxes_stacked(savepath, bbox_evaluator.step_outputs)
        # logger.info(args_dict)
        metrics = bbox_evaluator.evaluate(val_test)
        if args.name is not None:
            mlflow.log_metrics(metrics)

        # logger.info("Current args:")
        # logger.info(args.__dict__)
        logger.info("Metrics:")
        sorted_keys = sorted(list(metrics.keys()))
        logger.info("\n" + "\n".join([f"{key}: {metrics[key]}" for key in sorted_keys]))
    rand_suffix = get_random_file_name()
    path = Path(args.savedir) / f"metrics_{rand_suffix}.yml"
    logger.info(f"NOT Saving at: {path}")
    if args.name is not None:
        mlflow.end_run()


if __name__ == "__main__":
    main()
