# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import sys
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ecod.utils.general import ProgressLogger


class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _to_coco_format(bboxs_gt, labels_gt, bboxs_pred, labels_pred, scores_pred, height=240, width=304):
    """
    Adapted from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox
    Predicted and ground truth boxes are top_left, bottom_right; Convert to COCO format
    """
    annotations = []
    results = []
    images = []
    names = ["bboxs_gt", "labels_gt", "bboxs_pred", "labels_pred", "scores_pred"]
    lens = {key: len(vv) for key, vv in zip(names, [bboxs_gt, labels_gt, bboxs_pred, labels_pred, scores_pred])}
    if any(ll != lens[list(lens.keys())[0]] for ll in lens.values()):
        raise RuntimeError(
            "Lengths between predictions and ground truth do not match; " f"There has to be a bug somewhere. {lens}"
        )
    # to dictionary
    it_counter = ProgressLogger(iterable_or_len=len(bboxs_pred), name="COCOCONVERT")
    im_id = 0
    for image_id, (
        bboxs_gt_i,
        labels_gt_i,
        bboxs_pred_i,
        labels_pred_i,
        scores_pred_i,
        _,
    ) in enumerate(zip(bboxs_gt, labels_gt, bboxs_pred, labels_pred, scores_pred, it_counter)):
        # skip empty gt samples that could be in here because we do the detection for a sequence instead of single-frame
        if len(bboxs_gt_i) == 0:
            continue
        im_id += 1
        images.append(
            {
                "date_captured": "2019",
                "file_name": "n.a",
                "id": im_id,
                "license": 1,
                "url": "",
                "height": height,
                "width": width,
            }
        )
        for bbox, label in zip(bboxs_gt_i, labels_gt_i):
            x1, y1 = bbox[:2]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            area = w * h
            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(label),
                "id": len(annotations) + 1,
            }
            annotations.append(annotation)

        for bbox, label, score in zip(bboxs_pred_i, labels_pred_i, scores_pred_i):
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            image_result = {
                "image_id": im_id,
                "category_id": int(label),
                "score": float(score),
                "bbox": [bbox[0], bbox[1], w, h],
            }
            results.append(image_result)

    dataset = {
        "info": {},
        "licenses": [],
        "type": "instances",
        "images": images,
        "annotations": annotations,
        "categories": None,
    }
    return dataset, results


def _calc_coco_metric(vals):
    vals = vals[vals > -1]
    if len(vals) > 0:
        return vals.mean()
    else:
        return -1


def coco_eval_metrics(coco_eval):
    n_classes = coco_eval.eval["counts"][2]
    iou50_idx = np.where(coco_eval.eval["params"].iouThrs == 0.5)[0]
    iou75_idx = np.where(coco_eval.eval["params"].iouThrs == 0.75)[0]

    evals_per_class = []
    for cl in range(n_classes):
        metrics = []
        metrics.append(_calc_coco_metric(coco_eval.eval["precision"][..., cl, 0, 2]))
        metrics.append(_calc_coco_metric(coco_eval.eval["precision"][iou50_idx, :, cl, 0, 2]))
        metrics.append(_calc_coco_metric(coco_eval.eval["precision"][iou75_idx, :, cl, 0, 2]))
        for ii in range(3):
            metrics.append(_calc_coco_metric(coco_eval.eval["precision"][..., cl, ii + 1, 2]))
        for ii in range(3):
            metrics.append(_calc_coco_metric(coco_eval.eval["recall"][..., cl, 0, ii]))
        for ii in range(3):
            metrics.append(_calc_coco_metric(coco_eval.eval["recall"][..., cl, ii + 1, 2]))
        evals_per_class.append(metrics)

    evals_tot = []
    evals_tot.append(_calc_coco_metric(coco_eval.eval["precision"][..., :, 0, 2]))
    evals_tot.append(_calc_coco_metric(coco_eval.eval["precision"][iou50_idx, :, :, 0, 2]))
    evals_tot.append(_calc_coco_metric(coco_eval.eval["precision"][iou75_idx, :, :, 0, 2]))
    for ii in range(3):
        evals_tot.append(_calc_coco_metric(coco_eval.eval["precision"][..., :, ii + 1, 2]))
    for ii in range(3):
        evals_tot.append(_calc_coco_metric(coco_eval.eval["recall"][..., :, 0, ii]))
    for ii in range(3):
        evals_tot.append(_calc_coco_metric(coco_eval.eval["recall"][..., :, ii + 1, 2]))
    # is this the same as above? Not sure because of excluding -1s
    # evals_per_class = np.array(evals_per_class)
    # evals_tot2 = []
    # for ii in range(12):
    #     evals_tot2.append(_calc_coco_metric(evals_per_class[:, ii]))
    return evals_tot, evals_per_class


# def coco_eval(bboxs_gt, labels_gt, bboxs_pred, labels_pred, scores_pred, height, width, labelmap=("car", "pedestrian")):
#    """simple helper function wrapping around COCO's Python API
#    :params:  gts iterable of numpy boxes for the ground truth
#    :params:  detections iterable of numpy boxes for the detections
#    :params:  height int
#    :params:  width int
#    :params:  labelmap iterable of class labels
#    """
#    import sys
#    import subprocess
#    dataset, results = _to_coco_format(bboxs_gt,
#                                       labels_gt,
#                                       bboxs_pred,
#                                       labels_pred,
#                                       scores_pred,
#                                       height=height,
#                                       width=width)
#    print("AUX-1", file=sys.stderr)
#    print(
#        float(
#            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
#                           stderr=subprocess.STDOUT).stdout.decode('utf-8').split("\n")[0]), "MB")
#    del bboxs_gt
#    del labels_gt
#    del bboxs_pred
#    del scores_pred
#    del labels_pred
#    import gc
#    gc.collect()
#    print("AUX0", file=sys.stderr)
#    print(
#        float(
#            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
#                           stderr=subprocess.STDOUT).stdout.decode('utf-8').split("\n")[0]), "MB")
#    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"} for id, class_name in enumerate(labelmap)]
#    dataset['categories'] = categories
#    #with HidePrints():
#    coco_results = _coco_stats(dataset, results)
#    return coco_eval_metrics(coco_results)


def coco_stats_names():
    return [
        "mAP",
        "AP50VOC",
        "AP75",
        "APsmall",
        "APmed",
        "APlarge",
        "AR1",
        "AR10",
        "AR100",
        "ARsmall",
        "ARmed",
        "ARlarge",
    ]


# def _coco_stats(dataset, results):
#    import sys
#    import subprocess
#    print("AUX1", file=sys.stderr)
#    cmd = "ps afu | awk 'NR>1 {$5=int($5/1024);}{ print;}' | grep offline_eval | grep python  | awk '{ print $5}'"
#    print(
#        float(
#            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
#                           stderr=subprocess.STDOUT).stdout.decode('utf-8').split("\n")[0]), "MB")
#    coco_gt = COCO()
#    coco_gt.dataset = dataset
#    coco_gt.createIndex()
#    coco_pred = coco_gt.loadRes(results)
#    del results
#    print("AUX5", file=sys.stderr)
#    print(
#        float(
#            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
#                           stderr=subprocess.STDOUT).stdout.decode('utf-8').split("\n")[0]), "MB")
#
#    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
#    coco_eval.params.imgIds = np.arange(1, len(dataset['images']) + 1, dtype=int)
#    coco_eval.evaluate()
#    print("AUX9", file=sys.stderr)
#    print(
#        float(
#            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
#                           stderr=subprocess.STDOUT).stdout.decode('utf-8').split("\n")[0]), "MB")
#    coco_eval.accumulate()
#    print("AUX10", file=sys.stderr)
#    coco_eval.summarize()
#    return coco_eval
