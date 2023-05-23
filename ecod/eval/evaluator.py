# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import ABC
import logging
from pathlib import Path
import pickle as pkl
import numpy as np
import h5py
import pandas as pd
from collections import OrderedDict

import torch
from torchvision.ops import box_iou, box_area

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from evis.trans import count_events_in_boxes

from ecod.data.proph1mpx.toolbox.psee_loader import PSEELoader
from ecod.data.box2d.transforms import normalized_boxes_to_absolute
from ecod.models.boxmemory import BoxMemory, count_events_in_boxes2
from ecod.eval.postprocessing import get_post_processor
from ecod.eval.stackedarray import (
    StackedArrayCat,
    StackedArrayEL,
    StackedArrayNEL,
    PreallocArrayNEL,
)
from ecod.eval.coco import (
    coco_stats_names,
    _to_coco_format,
    coco_eval_metrics,
    HidePrints,
)
from ecod.utils.data import get_dataset_attributes
from ecod.utils.files import load_args_dict
from ecod.utils.general import ProgressLogger
from ecod.training.callbacks import MemoryLogger
from ecod.paths import random_move_mnist36_root, random_move_debug_root
from ecod.data.rmmnist.dataset import RandomMoveMnistSingleIndexer
from ecod.data.proph1mpx.dataset import Prophesee1MpxSingleIndexer
from ecod.data.sim.rmmnist import load_rmmnist_events
from ecod.data.proph1mpx.load import proph_1mpx_events_to_events
from ecod.plot.boxmemory import rescale_boxes


def indexed_boxes_to_list(bboxs, labels, idxs, sample_idxs, n_timesteps):
    """Convert np.array(n_boxes, 4) to [bboxs_sample for _ in range(n_samples)]

    Args:
        bboxs (np.ndarray): shape (n_boxes, 4)
        labels (np.ndarray): shape (n_boxes, )
        idxs (np.ndarray): shape (n_boxes, 2), where idxs[:, 0] is the batch index and idxs[:, 1] is the index of the
            sequence in each batch. Example: idxs = np.array([[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]) are
            two batches, the first has a sequence of length 2, where two boxes are in the first part of the sequence
            and one box is in the second part of the sequence; in batch two, there is one box per sequence, and a
            sequence of length 3

    Returns:
        tuple(bboxs_list, labels_list): Converted bboxs and labels
    """
    bboxs = bboxs.cpu().numpy()
    labels = labels.cpu().numpy()
    idxs = idxs.cpu().numpy()
    sample_idxs = sample_idxs.cpu().numpy()
    idxs_un = np.unique(idxs[:, 0])
    bboxs_list = []
    labels_list = []
    idxs_out = []
    for sample_idx in sample_idxs:
        if sample_idx in idxs_un:
            idx_range = np.where(idxs[:, 0] == sample_idx)[0]
            assert (np.diff(idx_range) == 1).all(), f"Found mixed up sequence when indexing bboxs: {idxs}"
            start, stop = idx_range[0], idx_range[-1] + 1
            bboxs_seq = bboxs[start:stop]
            labels_seq = labels[start:stop]
            idxs_seq = idxs[start:stop][:, 1]
            idxs_seq_un = np.unique(idxs_seq)
            for seq_idx in range(n_timesteps):
                if seq_idx in idxs_seq_un:
                    idx_seq_range = np.where(idxs_seq == seq_idx)[0]
                    assert (np.diff(idx_seq_range) == 1).all(), f"Found mixed up sequence when indexing bboxs: {idxs}"
                    start_seq, stop_seq = idx_seq_range[0], idx_seq_range[-1] + 1
                    bboxs_list.append(bboxs_seq[start_seq:stop_seq])
                    labels_list.append(labels_seq[start_seq:stop_seq])
                else:
                    bboxs_list.append(np.zeros((0, 4)))
                    labels_list.append(np.zeros((0,)))
                idxs_out.append(np.array([sample_idx, seq_idx]))
        else:
            bboxs_list += [np.zeros((0, 4))] * n_timesteps
            labels_list += [np.zeros((0,))] * n_timesteps
            idxs_out += [np.array([sample_idx, -1])] * n_timesteps
    return bboxs_list, labels_list, idxs_out


def save_predicted_boxes(savepath, results):
    with h5py.File(savepath, "w") as hd:
        hd.attrs["len"] = len(results["labels_gt"])
        load_idxs = {key: np.insert(np.cumsum([len(ll) for ll in results[key]]), 0, 0) for key in results.keys()}
        for key, array_list in results.items():
            hd.create_dataset(f"{key}/load_idxs", data=load_idxs[key], compression=None)
            hd.create_dataset(f"{key}/data", data=np.concatenate(array_list), compression=None)


def load_predicted_boxes(path):
    with h5py.File(path, "r") as hd:
        results = {key: [] for key in hd}
        for key in hd:
            array = hd[f"{key}/data"][:]
            lis = hd[f"{key}/load_idxs"][:]
            array_list = [array[lis[ii] : lis[ii + 1]] for ii in range(len(lis) - 1)]
            results[key] = array_list
    return results


def save_predicted_boxes_stacked(savepath, results):
    with h5py.File(savepath, "w") as hd:
        for key, stacked_array in results.items():
            if len(stacked_array.stack) != 1:
                stacked_array.accumulate_stack()
            if type(stacked_array) == StackedArrayNEL:
                load_idxs = results[key].cum_lens
                hd.create_dataset(f"{key}/load_idxs", data=load_idxs, compression=None)
                hd.create_dataset(f"{key}/data", data=stacked_array.stack[0], compression="lzf")
            elif type(stacked_array) == PreallocArrayNEL:
                load_idxs = results[key].cum_lens
                hd.create_dataset(f"{key}/load_idxs", data=load_idxs, compression=None)
                hd.create_dataset(f"{key}/data", data=stacked_array.array, compression="lzf")
            elif type(stacked_array) == StackedArrayEL:
                dset = hd.create_dataset(f"{key}/data", data=stacked_array.stack[0], compression="lzf")
                dset.attrs["stacked_array_type"] = "EL"
            elif type(stacked_array) == StackedArrayCat:
                dset = hd.create_dataset(f"{key}/data", data=stacked_array.stack[0], compression="lzf")
                dset.attrs["stacked_array_type"] = "Cat"
            else:
                raise TypeError(f"This object should be a StackedArray but is: {type(stacked_array)}")
            if key == "labels_gt":
                hd.attrs["len"] = len(stacked_array)


def load_predicted_boxes_stacked(path):
    with h5py.File(path, "r") as hd:
        results = {}
        for key in hd:
            stacked_type = hd[key].attrs.get("stacked_array_type", None)
            array = hd[f"{key}/data"][:]
            if "load_idxs" in hd[key]:
                lis = hd[f"{key}/load_idxs"][:]
                data = StackedArrayNEL.from_array(array, lis, name=key)
                # patch loading file_idxs correctly for older experiments
            elif key == "file_idxs":
                data = StackedArrayCat.from_array(array, name=key)
            elif stacked_type == "EL":
                data = StackedArrayEL.from_array(array, len(array), name=key)
            elif stacked_type == "Cat":
                data = StackedArrayCat.from_array(array, name=key)
            else:
                # fix missing 'Cat' key from early recordings of frame_times_ms here
                if key == "frame_times_ms":
                    data = StackedArrayCat.from_array(array, name=key)
                else:
                    raise RuntimeError(f"Tried to load predicted boxes but could not identify type of {key}")
            results[key] = data
    return results


def get_predicted_box_data(artifact_uri, name):
    name = f"best_{name}_bboxs.h5" if name == "val" else f"{name}_bboxs.h5"
    box_path = artifact_uri / name
    if not box_path.exists():
        box_path_pkl = box_path.with_suffix(".pkl")
        if not box_path_pkl.exists():
            raise RuntimeError(f"The box path at {box_path} and {box_path_pkl} do not exist")
        with open(box_path, "rb") as hd:
            box_data = pkl.load(hd)
    else:
        box_data = load_predicted_boxes_stacked(box_path)
    return box_data, box_path


class EventsLoader:
    def __init__(self, hparams, val_test="val"):
        self.hparams = hparams
        self.val_test = val_test
        self.dset = hparams["dataset"]
        self.current_file_idx = -1
        self.current_t_ms_start_stop = np.array([-1, -1])
        self.current_t_idxs = np.array([-1, -1])
        self.events = None
        self.root = None
        self.sorted_paths = self.get_file_paths()

    def get_file_paths(self):
        if self.dset.startswith("random_"):
            if self.dset == "random_move_mnist36_od":
                self.root = random_move_mnist36_root
            elif self.dset == "random_move_debug_od":
                self.root = random_move_debug_root
            else:
                raise ValueError(f"dset {self.dset} not implemented")
            indexer = RandomMoveMnistSingleIndexer(
                self.root,
                self.hparams["bbox_suffix_test"],
                train_val_test=self.val_test,
            )
            sorted_paths = indexer.get_sorted_paths()
        elif self.dset.lower() in ["proph_1mpx", "proph_1mpx"]:
            indexer = Prophesee1MpxSingleIndexer(self.val_test, self.hparams["bbox_suffix_test"])
            sorted_paths = indexer.get_paths()
        return sorted_paths

    def load_events(self, file_idx, time_ms_start, time_ms_stop):
        if self.dset in ["random_move_mnist36_od", "random_move_debug_od"]:
            return self.load_events_rmmnist(file_idx, time_ms_start, time_ms_stop)
        elif self.dset.lower() in ["proph_1mpx", "proph_1mpx"]:
            return self.load_events_proph_1mpx(file_idx, time_ms_start, time_ms_stop)
        else:
            raise ValueError(f"self.dset has to be a valid dataset name, but is {self.dset}")


    def load_events_rmmnist(self, file_idx, time_ms_start, time_ms_stop):
        if file_idx != self.current_file_idx:
            self.events = load_rmmnist_events(self.root, file_idx, train_val_test=self.val_test, to_float=True)
            self.current_file_idx = file_idx
        idx_start, idx_stop = np.searchsorted(self.events[:, 0], [time_ms_start * 1e3, time_ms_stop * 1e3])
        return self.events[idx_start:idx_stop]

    def load_events_proph_1mpx(self, file_idx, time_ms_start, time_ms_stop):
        t_start = time_ms_start * 1e3
        t_stop = time_ms_stop * 1e3
        if file_idx != self.current_file_idx:
            path = self.sorted_paths[file_idx]
            self.events = PSEELoader(str(path))
            self.current_file_idx = file_idx
            self.events.seek_time(t_start)
        elif self.current_t_ms_start_stop[1] - t_start > 1.6:
            raise RuntimeError(
                "Time between end of last sample and start of current is more than 1.6ms, " "should be close to 0"
            )
        events = self.events.load_delta_t(t_stop - t_start)
        return proph_1mpx_events_to_events(events, shift_time_to_zero=False)


def get_evaluator(hparams):
    name = hparams["evaluator_name"]
    if name == "standard":
        return StandardEvaluator(hparams)
    else:
        raise ValueError(f"evaluator_name has to be in 'standard', but is {name}")


class Evaluator(ABC):
    def __init__(self):
        raise NotImplementedError()

    def __call__(
        self,
        outs,
        seqs,
        bboxs_gt,
        labels_gt,
        idxs,
        sample_idxs,
        file_idxs,
        frame_times_ms,
    ):
        raise NotImplementedError()

    def accumulate_sample(self):
        raise NotImplementedError()

    def accumulate(self):
        raise NotImplementedError()

    def reset(self):
        self.post_processed = False
        self.step_outputs = {
            "bboxs_gt": StackedArrayNEL(name="bboxs_gt"),
            "labels_gt": StackedArrayNEL(name="labels_gt"),
            "idxs": StackedArrayNEL(name="idxs"),
            "file_idxs": StackedArrayCat(name="file_idxs"),
            "frame_times_ms": StackedArrayCat(name="frame_times_ms"),
        }
        if self.save_all_priors:
            self.step_outputs.update(
                {
                    "bboxs_pred": StackedArrayEL(name="bboxs_pred"),
                    "scores_pred": StackedArrayEL(name="scores_pred"),
                }
            )
        else:
            self.step_outputs.update(
                {
                    "bboxs_pred": StackedArrayNEL(name="bboxs_pred"),
                    "scores_pred": StackedArrayNEL(name="scores_pred"),
                }
            )

    def overwrite_outputs(self, outputs):
        allowed_keys = [
            "bboxs_gt",
            "labels_gt",
            "idxs",
            "file_idxs",
            "bboxs_pred",
            "scores_pred",
            "labels_pred",
        ]
        for key in outputs.keys():
            if key in ["frame_times_ms", "file_idxs_dset"]:
                continue
            if key not in allowed_keys:
                raise ValueError(f"output keys have to be in {allowed_keys}, but found: {key}")
            if type(outputs[key]) not in [
                StackedArrayNEL,
                StackedArrayEL,
                PreallocArrayNEL,
                StackedArrayCat,
            ]:
                raise TypeError(
                    f"type of overwrite outputs has to be StackedArray EL/NEL/Cat but is " f"{type(outputs[key])}"
                )
            self.step_outputs[key] = outputs[key]
        if outputs.get("frame_times_ms", None) is not None:
            if type(outputs["frame_times_ms"]) != StackedArrayCat:
                self.check_dset_outputs(outputs)
                self.step_outputs["frame_times_ms"] = StackedArrayCat.from_array(outputs["frame_times_ms"])
                self.step_outputs["file_idxs_dset"] = StackedArrayCat.from_array(outputs["file_idxs_dset"])
            else:
                self.step_outputs["frame_times_ms"] = outputs["frame_times_ms"]

    def check_dset_outputs(self, outputs):
        file_idxs_dset = outputs["file_idxs_dset"]
        file_idxs_data = outputs["file_idxs"].array
        if file_idxs_dset.shape != file_idxs_data.shape:
            raise RuntimeError(
                "Shapes of file_idxs_data and file_idxs_dset don't match: " f"{file_idxs_data} vs {file_idxs_dset}"
            )
        mask = file_idxs_dset != file_idxs_data
        if mask.any():
            raise RuntimeError(
                "file_idxs_data and file_idxs_dset are not exactly the same: "
                f"{file_idxs_data[mask]} vs {file_idxs_dset[mask]}"
            )


class StandardEvaluator(Evaluator):
    def __init__(self, hparams, save_all_priors=False, confidence_thresh_save=0.01):
        self.hparams = hparams
        self.savedir = hparams["temp_dir"]
        self.save_all_priors = save_all_priors
        self.dataset_attrs = get_dataset_attributes(hparams["dataset"])
        self.post_processor = get_post_processor(hparams)
        self.logger = logging.getLogger("EVAL")
        self.logger.setLevel("INFO")
        self.mem_logger = MemoryLogger("EVAL")
        self.step_outputs = None
        self.post_processed = None
        # all bboxs where background has a confidence above 1 - conf_thresh_save will be discarded. This saves a lot
        # (factor 10) of disk space, with minor changes in mAP (== 0 if test_confidence_thresh >= conf_thresh_save)
        self.confidence_threshold_for_save = confidence_thresh_save
        self.reset()

    def __call__(
        self,
        scores,
        bbox_reg_pred,
        seqs,
        bboxs_gt,
        labels_gt,
        idxs,
        sample_idxs,
        file_idxs,
        frame_times_ms,
    ):
        n_timesteps = seqs.shape[1]
        image_shape = seqs.shape[-2:]
        step_output = self.accumulate_sample(
            scores,
            bbox_reg_pred,
            bboxs_gt,
            labels_gt,
            idxs,
            sample_idxs,
            file_idxs,
            frame_times_ms,
            n_timesteps,
            image_shape,
        )
        for key, val in step_output.items():
            self.step_outputs[key](val)

    def accumulate_sample(
        self,
        scores,
        bbox_reg_pred,
        bboxs_gt,
        labels_gt,
        idxs,
        sample_idxs,
        file_idxs,
        frame_times_ms,
        n_timesteps,
        image_shape,
    ):
        bboxs_gt = normalized_boxes_to_absolute(bboxs_gt, image_shape, copy=False)
        bboxs, labels, idxs = indexed_boxes_to_list(
            bboxs_gt, labels_gt, idxs, sample_idxs=sample_idxs, n_timesteps=n_timesteps
        )
        if self.save_all_priors:
            bboxs_pred, scores_pred = [ss.cpu().numpy().reshape(-1, *ss.shape[2:]) for ss in [bbox_reg_pred, scores]]
        else:
            # bboxs_pred, scores_pred = self.select_one_class_per_prior(scores, bbox_reg_pred)
            bboxs_pred, scores_pred = self.select_high_threshold_boxes(scores, bbox_reg_pred)
        file_idxs = file_idxs.repeat_interleave(n_timesteps)
        outputs = {
            "bboxs_gt": bboxs,
            "labels_gt": labels,
            "idxs": idxs,
            "file_idxs": file_idxs.cpu().numpy(),
            "bboxs_pred": bboxs_pred,
            "scores_pred": scores_pred,
            "frame_times_ms": frame_times_ms[:-1].cpu().numpy(),
        }
        self.check_outputs(outputs)
        return outputs

    def select_high_threshold_boxes(self, scores, bbox_reg_pred):
        scores = scores.reshape(-1, *scores.shape[2:]).cpu().numpy()
        bbox_reg_pred = bbox_reg_pred.reshape(-1, *bbox_reg_pred.shape[2:]).cpu().numpy()
        mask = scores[:, :, 0] < 1 - self.confidence_threshold_for_save
        scores_pred = []
        bboxs_pred = []
        for ii in range(len(mask)):
            scores_pred.append(scores[ii][mask[ii]])
            bboxs_pred.append(bbox_reg_pred[ii][mask[ii]])
        return bboxs_pred, scores_pred

    def select_one_class_per_prior(self, scores, bbox_reg_pred):
        labels_pred = torch.argmax(scores, 3)
        # (b, t, p) -> (b*t, p)
        scores = scores.reshape(scores.shape[0] * scores.shape[1], *scores.shape[2:])
        bbox_reg_pred = bbox_reg_pred.reshape(bbox_reg_pred.shape[0] * bbox_reg_pred.shape[1], *bbox_reg_pred.shape[2:])
        labels_pred = labels_pred.reshape(labels_pred.shape[0] * labels_pred.shape[1], *labels_pred.shape[2:])
        bboxs_pred, scores_pred, labels_pred = [ss.cpu().numpy() for ss in [bbox_reg_pred, scores, labels_pred]]
        mask = labels_pred > 0
        bboxs_pred = [bboxs[mask[ii]] for ii, bboxs in enumerate(bboxs_pred)]
        scores_pred = [scores_single[mask[ii]] for ii, scores_single in enumerate(scores_pred)]
        return bboxs_pred, scores_pred

    @staticmethod
    def check_outputs(outputs):
        lens = {key: len(vv) for key, vv in outputs.items()}
        if any(ll != lens[list(lens.keys())[0]] for ll in lens.values()):
            raise RuntimeError(
                "Lengths between predictions and ground truth do not match; "
                f"There has to be a bug somewhere. {lens}, {outputs['idxs']}"
            )
        # TODO: I don't think we have to do this anymore, because we handle it in the eval step; but check back if
        # there are any problems
        pop_empty = False
        if pop_empty:
            idxs_pop = []
            for ii, oo in enumerate(outputs["bboxs_gt"]):
                # len == 0 -> have to remove prediction; this is necessary because we predict boxes every time step,
                # but for some time steps there are no ground truth boxes
                if len(oo) == 0:
                    idxs_pop.append(ii)
            for ii in idxs_pop[::-1]:
                for key in outputs.keys():
                    outputs[key].pop(ii)

    def remove_results_with_empty_gt_boxes(self):
        mask = self.step_outputs["bboxs_gt"].zero_elements_mask()
        for key in [
            "bboxs_gt",
            "labels_gt",
            "bboxs_pred",
            "labels_pred",
            "scores_pred",
        ]:
            self.step_outputs[key].remove_zeros_with_mask(mask)

    def coco_eval(self, val_test):
        labelmap = self.dataset_attrs["class_names"][1:]
        height, width = self.dataset_attrs["shape_max"]

        self.remove_results_with_empty_gt_boxes()
        results = self.step_outputs
        self.logger.info("Starting COCO evaluation")
        self.logger.info("Data preparation")
        self.mem_logger.log()
        dataset, results = _to_coco_format(
            results["bboxs_gt"],
            results["labels_gt"],
            results["bboxs_pred"],
            results["labels_pred"],
            results["scores_pred"],
            height=height,
            width=width,
        )
        self.step_outputs = None
        categories = [
            {"id": id + 1, "name": class_name, "supercategory": "none"} for id, class_name in enumerate(labelmap)
        ]
        dataset["categories"] = categories
        with HidePrints():
            self.logger.info("Indexing coco ground truth dataset")
            self.mem_logger.log()
            coco_gt = COCO()
            coco_gt.dataset = dataset
            coco_gt.createIndex()
            self.logger.info("Indexing predictions")
            self.mem_logger.log()
            coco_pred = coco_gt.loadRes(results)
            results = None

            coco_evaluator = COCOeval(coco_gt, coco_pred, "bbox")
            coco_gt = None
            coco_pred = None
            # coco_evaluator.params.imgIds = np.arange(1, len(dataset['images']) + 1, dtype=int)
            self.logger.info("Evaluating each sample")
            self.mem_logger.log()
            coco_evaluator.evaluate()
            dataset = None
            self.logger.info("Accumulating results")
            self.mem_logger.log()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        eval_tot, eval_per_class = coco_eval_metrics(coco_evaluator)
        coco_names = coco_stats_names()
        metric_names = [f"{val_test}_{coco_names[0]}"] + [f"_{val_test}_{name}" for name in coco_names[1:]]
        metrics = {name: np.float32(met) for name, met in zip(metric_names, eval_tot)}
        for label_name, eval_c in zip(labelmap, eval_per_class):
            for ii, name in enumerate(coco_names):
                if ii == 0:
                    metrics[f"{val_test}_{name}_{label_name}"] = eval_c[ii]
                else:
                    metrics[f"_{val_test}_{name}_{label_name}"] = eval_c[ii]
        return metrics

    def save_predicted_boxes(self, results, name, as_pkl=False):
        self.logger.info("Starting saving predicted boxes")
        if as_pkl:
            path = Path(self.savedir) / f"{name}_bboxs.pkl"
            with open(path, "wb") as hd:
                pkl.dump(results, hd)
        else:
            path = Path(self.savedir) / f"{name}_bboxs.h5"
            save_predicted_boxes_stacked(path, results)
        self.logger.info("Finished saving predicted boxes")
        return path

    def delete_range(self, process_only_n_percent=None, idx_start=0):
        if process_only_n_percent is not None:
            idx_max = int(len(self.step_outputs["scores_pred"]) * process_only_n_percent) + idx_start
            self.step_outputs = {
                key: self.step_outputs[key].delete_range(idx_start, idx_max) for key in self.step_outputs.keys()
            }
        return None

    def post_process_all(self):
        results = dict(
            bboxs_pred=StackedArrayNEL(name="bboxs_pred"),
            labels_pred=StackedArrayNEL(name="labels_pred"),
            scores_pred=StackedArrayNEL(name="scores_pred"),
        )
        key_map = {
            "boxes": "bboxs_pred",
            "labels": "labels_pred",
            "scores": "scores_pred",
        }
        key_map_rev = {val: key for key, val in key_map.items()}
        self.logger.info("Starting post-processing")
        step_outputs = self.step_outputs
        for idx in ProgressLogger(range(len(step_outputs["scores_pred"])), name="POSTPROC"):
            scores_pred = step_outputs["scores_pred"][idx]
            scores_pred = torch.from_numpy(scores_pred[None])
            bboxs_pred = step_outputs["bboxs_pred"][idx]
            bboxs_pred = torch.from_numpy(bboxs_pred[None])
            detections = (scores_pred, bboxs_pred)
            res_single = self.post_processor(detections)
            if len(res_single) != 1:
                raise RuntimeError(f"Something is wrong here; Batch size should be 1 but is {len(res_single)}")
            for key in results.keys():
                results[key](res_single[0][key_map_rev[key]].numpy())
            # step_outputs['scores_pred'].delete(0)
            # step_outputs['bboxs_pred'].delete(0)
        self.logger.info("Finished post-processing")
        return results

    def add_boxes_from_memory(
        self,
        val_test,
        box_mem_conf_thresh=0.3,
        box_mem_forget_thresh=0.00001,
        box_mem_delete_thresh=100,
        box_mem_iou_thresh=0.5,
        box_mem_savepath=None,
        box_mem_load_counts=False,
        box_mem_plot=False,
    ):
        events_loader = EventsLoader(self.hparams, val_test)
        events_shape = self.dataset_attrs["shape_max"]
        box_shape = self.hparams["shape_t"][-2:]
        box_memory = BoxMemory(
            self.hparams,
            events_shape,
            box_shape,
            box_mem_conf_thresh,
            box_mem_forget_thresh,
            box_mem_delete_thresh,
            iou_thresh=box_mem_iou_thresh,
            savepath=box_mem_savepath,
            load_counts=box_mem_load_counts,
            plot=box_mem_plot,
        )
        dt_ms = self.dataset_attrs["delta_t_mus"] / 1e3
        keys = ["bboxs_pred", "labels_pred", "scores_pred"]
        results = {key: StackedArrayNEL(name=key) for key in keys}
        for idx in ProgressLogger(range(len(self.step_outputs["scores_pred"])), name="BOXMEM"):
            time_ms_start = self.step_outputs["frame_times_ms"][idx]
            time_ms_stop = time_ms_start + dt_ms
            file_idx = self.step_outputs["file_idxs"][idx]
            if hasattr(file_idx, "__len__"):
                raise RuntimeError(f"Somehow there is more than one file idx at {idx}: {file_idx}")
            events = events_loader.load_events(file_idx, time_ms_start, time_ms_stop)
            # event_frame = np.array(events_to_frame_2c(events, events_max_size[0], events_max_size[1]))
            boxes_gt = self.step_outputs["bboxs_gt"][idx]
            labels_gt = self.step_outputs["labels_gt"][idx]
            boxes = self.step_outputs["bboxs_pred"][idx]
            scores = self.step_outputs["scores_pred"][idx]
            labels = self.step_outputs["labels_pred"][idx]
            boxes, labels, scores = box_memory(boxes, labels, scores, events, file_idx, boxes_gt, labels_gt)
            results["bboxs_pred"](boxes)
            results["labels_pred"](labels)
            results["scores_pred"](scores)
        self.overwrite_outputs(results)
        self.logger.info(f"BOXMEMORY stats: {box_memory.stats}")

    def stats_tp_or_fn_per_box(
        self,
        val_test,
        conf_thresh=0.3,
        iou_thresh=0.5,
    ):
        events_loader = EventsLoader(self.hparams, val_test)
        events_shape = self.dataset_attrs["shape_max"]
        box_shape = self.hparams["shape_t"][-2:]
        scale_factors_xy = np.array([events_shape[ii] / box_shape[ii] for ii in range(2)][::-1])
        dt_ms = self.dataset_attrs["delta_t_mus"] / 1e3
        results = []
        for idx in ProgressLogger(range(len(self.step_outputs["scores_pred"])), name="TPFN"):
            time_ms_start = self.step_outputs["frame_times_ms"][idx]
            time_ms_stop = time_ms_start + dt_ms
            file_idx = self.step_outputs["file_idxs"][idx]
            if hasattr(file_idx, "__len__"):
                raise RuntimeError(f"Somehow there is more than one file idx at {idx}: {file_idx}")
            events = events_loader.load_events(file_idx, time_ms_start, time_ms_stop)
            # event_frame = np.array(events_to_frame_2c(events, events_max_size[0], events_max_size[1]))
            boxes_gt = self.step_outputs["bboxs_gt"][idx]
            labels_gt = self.step_outputs["labels_gt"][idx]
            boxes = self.step_outputs["bboxs_pred"][idx]
            scores = self.step_outputs["scores_pred"][idx]
            labels = self.step_outputs["labels_pred"][idx]
            mask_score = scores > conf_thresh
            boxes, labels, scores = (
                boxes[mask_score],
                labels[mask_score],
                scores[mask_score],
            )
            boxes_scaled = rescale_boxes(boxes_gt, scale_factors_xy)
            areas = box_area(torch.from_numpy(boxes_scaled)).numpy()
            counts = count_events_in_boxes(events, boxes_scaled)
            # for each gt box: was detected (iou > thresh) and event count
            for cc in np.unique(labels_gt):
                idxs_gt = np.nonzero(labels_gt == cc)[0]
                idxs_pred = np.nonzero(labels == cc)[0]
                boxes_gt_cc = boxes_gt[idxs_gt]
                if len(idxs_pred) == 0:
                    is_true_positive = np.full((len(boxes_gt_cc)), False)
                else:
                    ious = box_iou(
                        torch.from_numpy(boxes_gt_cc),
                        torch.from_numpy(boxes[idxs_pred]),
                    )
                    is_true_positive = []
                    # have to set ious to -1 because don't want that one predicted box has multiple true positives, can
                    # have 1 at most (because close objects standing next to each other have to be detected separately)
                    for ii in range(len(ious)):
                        ious_per_pred = ious[ii]
                        top_iou_idx = ious_per_pred.argmax()
                        iou_max = ious_per_pred[top_iou_idx]
                        if iou_max > iou_thresh:
                            is_true_positive.append(True)
                            ious[:, top_iou_idx] = -1
                        else:
                            is_true_positive.append(False)
                    is_true_positive = np.array(is_true_positive)
                label_cc = np.full((len(boxes_gt_cc)), cc)
                counts_cc = counts[idxs_gt]
                areas_cc = areas[idxs_gt]
                result = np.c_[label_cc, areas_cc, counts_cc, is_true_positive]
                results.append(result)
        dtypes = OrderedDict(
            [
                ("label", np.int32),
                ("area", np.float32),
                ("count", np.int32),
                ("is_tp", bool),
            ]
        )
        df = pd.DataFrame(np.concatenate(results, axis=0), columns=list(dtypes.keys())).astype(dtypes)
        return df

    def save_and_evaluate(self, val_test):
        path = self.save_predicted_boxes(self.step_outputs, val_test)
        metrics = self.evaluate(val_test)
        return path, metrics

    def evaluate(self, val_test):
        if not self.post_processed:
            results_pred = self.post_process_all()
            for key, val in results_pred.items():
                self.step_outputs[key] = val
            self.post_processed = True
        metrics = self.coco_eval(val_test)
        return metrics


def load_and_postprocess(
    expdir,
    val_test,
    args_dict=None,
    postprocessor_name=None,
    test_confidence_threshold=None,
    test_nms_threshold=None,
    test_max_per_image=None,
    process_only_n_percent=None,
    box_mem=True,
    box_mem_conf_thresh=0.3,
    box_mem_forget_thresh=0.0001,
    box_mem_delete_thresh=100,
    box_mem_iou_thresh=0.5,
    box_mem_savepath=None,
    box_mem_load_counts=False,
    box_mem_plot=False,
    idx_start=0,
):
    overwrite_args = {
        "postprocessor_name": postprocessor_name,
        "test_confidence_threshold": test_confidence_threshold,
        "test_nms_threshold": test_nms_threshold,
        "test_max_per_image": test_max_per_image,
    }
    if args_dict is None:
        args_dict = load_args_dict(expdir)
    for key, arg in overwrite_args.items():
        if arg is not None:
            args_dict[key] = arg
    bbox_data, _ = get_predicted_box_data(expdir, val_test)
    if bbox_data.get("frame_times_ms", None) is None:
        from ecod.data.lightning import SeqODDataModule

        dmod = SeqODDataModule(args_dict)
        dmod.prepare_data()
        dmod.setup()
        if val_test == "val":
            _, dset, _ = dmod.select_dset()
        elif val_test == "test":
            _, _, dset = dmod.select_dset()
        dmod = None
        times_ms = []
        file_idxs = []
        for indexer_idx in range(len(dset.indexer)):
            if args_dict["dataset"].startswith("random"):
                file_idx, _, frame_times_ms, _ = dset.indexer(indexer_idx)
            else:
                file_idx, frame_times_ms, _, _ = dset.indexer(indexer_idx)
            times_ms.append(frame_times_ms[:-1])
            file_idxs.append([file_idx] * (len(frame_times_ms) - 1))
        times_ms = np.concatenate(times_ms)
        file_idxs = np.concatenate(file_idxs)
        bbox_data["frame_times_ms"] = times_ms
        bbox_data["file_idxs_dset"] = file_idxs
    bbox_evaluator = get_evaluator(args_dict)
    bbox_evaluator.overwrite_outputs(bbox_data)
    bbox_evaluator.delete_range(process_only_n_percent=process_only_n_percent, idx_start=idx_start)
    bbox_evaluator.overwrite_outputs(bbox_evaluator.post_process_all())
    bbox_evaluator.post_processed = True
    if box_mem:
        bbox_evaluator.add_boxes_from_memory(
            val_test,
            box_mem_conf_thresh=box_mem_conf_thresh,
            box_mem_forget_thresh=box_mem_forget_thresh,
            box_mem_delete_thresh=box_mem_delete_thresh,
            box_mem_iou_thresh=box_mem_iou_thresh,
            box_mem_savepath=box_mem_savepath,
            box_mem_load_counts=box_mem_load_counts,
            box_mem_plot=box_mem_plot,
        )
    return bbox_evaluator, args_dict
