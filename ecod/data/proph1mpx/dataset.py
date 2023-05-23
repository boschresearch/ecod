# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import h5py
import numpy as np

import torch
from torchvision.ops import box_convert

from ecod.data.proph1mpx.toolbox.psee_loader import PSEELoader
from ecod.utils.data import get_dataset_attributes, identity_transform
from ecod.utils.general import ProgressLogger
from ecod.utils.files import makedirs
from ecod.paths import proph_1mpx_aux_data_path, proph_1mpx_paths
from ecod.data.proph1mpx.load import (
    load_boxes,
    box_path_from_events_path,
    proph_1mpx_events_to_events,
    proph_1mpx_boxes_to_array,
)
from ecod.data.transforms import SSDTargetTransform, SSDTargetBackTransform, VoxelGridTransform
from ecod.events.voxel import events_to_seq_voxel_grid_2c
from ecod.data.box2d.priors import PriorBox
from ecod.data.box2d.transforms import normalize_boxes


class Prophesee1MpxSingleIndexer:
    def __init__(self, train_val_test="train", bbox_suffix="filtered"):
        self.root = Path(proph_1mpx_paths[train_val_test]).parent
        self.tvt = train_val_test
        self.bbox_suffix = bbox_suffix
        self.paths = self.get_paths()
        indices_path = self.get_indices_path()
        if indices_path.exists():
            (
                self.file_idx_to_dset_idx,
                self.dset_idx_max_to_file_idx,
            ) = self.load_indices()
        else:
            if not indices_path.parent.exists():
                makedirs(indices_path.parent, overwrite=False)
            self.file_idx_to_dset_idx, self.dset_idx_max_to_file_idx = self.index_dset()
            self.save_indices()
        # self.file_idx_to_dset_idx, self.dset_idx_max_to_file_idx = self.index_dset()
        self.len = max(self.dset_idx_max_to_file_idx)

    def get_paths(self):
        sorted_paths = sorted([str(aa) for aa in proph_1mpx_paths[self.tvt].glob("*.dat")])
        return sorted_paths

    def index_dset(self):
        file_idx_to_dset_idx = {}
        dset_idx_max_to_file_idx = []
        sorted_paths = self.paths
        idx_glob = 0
        for idx_file, pp in enumerate(ProgressLogger(sorted_paths, f"idx_{self.tvt}")):
            boxes_handle = PSEELoader(box_path_from_events_path(pp, bbox_suffix=self.bbox_suffix))
            boxes = load_boxes(boxes_handle)
            times = np.unique(boxes["t"])
            n_items = len(times)
            glob_idxs = np.arange(n_items) + idx_glob
            file_idx_to_dset_idx[idx_file] = np.c_[glob_idxs, times]
            idx_glob += n_items
            dset_idx_max_to_file_idx.append(idx_glob - 1)
        return file_idx_to_dset_idx, dset_idx_max_to_file_idx

    def get_indices_path(self):
        return proph_1mpx_aux_data_path / self.bbox_suffix / f"index_map_{self.tvt}.h5"

    def save_indices(self):
        savepath = self.get_indices_path()
        if savepath.exists():
            savepath.unlink()
        with h5py.File(savepath, "w") as hd:
            for idx, arr in self.file_idx_to_dset_idx.items():
                hd.create_dataset(f"{idx:0>5}", data=arr, dtype=np.uint32)
            hd.create_dataset(f"map", data=self.dset_idx_max_to_file_idx, dtype=np.uint32)

    def load_indices(self):
        path = self.get_indices_path()
        dset_idx_max_to_file_idx = []
        file_idx_to_dset_idx = {}
        with h5py.File(path, "r") as hd:
            keys = [kk for kk in hd.keys() if kk != "map"]
            dset_idx_max_to_file_idx = hd["map"][:]
            for key in keys:
                file_idx_to_dset_idx[int(key)] = hd[key][:]
        return file_idx_to_dset_idx, dset_idx_max_to_file_idx

    def __call__(self, idx):
        file_idx = np.searchsorted(self.dset_idx_max_to_file_idx, idx)
        time_idx = np.searchsorted(self.file_idx_to_dset_idx[file_idx][:, 0], idx)
        bbox_time = self.file_idx_to_dset_idx[file_idx][time_idx, 1]
        return file_idx, time_idx, bbox_time

    def __len__(self):
        return self.len


class Prophesee1MpxSeqIndexer:
    def __init__(
        self,
        n_seqs,
        train_val_test="train",
        pre_index_events=True,
        bbox_suffix="filtered",
    ):
        self.n_seqs = n_seqs
        self.tvt = train_val_test
        self.bbox_suffix = bbox_suffix
        self.indexer = Prophesee1MpxSingleIndexer(self.tvt, bbox_suffix=bbox_suffix)
        attrs = get_dataset_attributes("proph_1mpx")
        self.delta_t_mus = attrs["delta_t_mus"]
        self.dt_var_bboxs_mus = 0.49 * self.delta_t_mus
        self.dt_var_test_times_mus = 0.2 * self.delta_t_mus
        self.pre_index_events = pre_index_events
        self.box_times = self.get_box_times()
        self.idx_to_seq_idx_map = self.get_idx_to_seq_idx_map()
        self.len = len(self.indexer) if self.tvt == "train" else len(self.idx_to_seq_idx_map)
        if pre_index_events:
            if self.get_event_indices_path().exists():
                self.event_indices = self.load_indices()
            else:
                self.event_indices = self.index_events()
                self.save_indices()

    def get_event_indices_path(self):
        path = self.indexer.get_indices_path().parent / f"index_map_{self.tvt}_{self.n_seqs:0>3}.h5"
        return path

    def index_events(self):
        event_start_stops = []
        delta_search_mus = 1e6
        path = self.indexer.paths[0]
        file_idx_last = 0
        events_handle = PSEELoader(str(path))
        for ii in ProgressLogger(range(self.len), name=f"IDX_{self.tvt.upper()}", every_n_percent=1.0):
            file_idx, t_range = self.get_t_range(ii)
            if file_idx != file_idx_last:
                path = self.indexer.paths[file_idx]
                events_handle = PSEELoader(str(path))
            diff = events_handle.current_time - t_range[0]
            if np.abs(diff) > delta_search_mus:
                events_handle.seek_time(t_range[0])
            elif diff < 0:
                events_handle.seek_future_time(t_range[0])
            else:
                events_handle.seek_past_time(t_range[0])
            p_start = events_handle.current_event
            # will have first case in almost all samples; just if there are no events for some time, can happen that
            # current_time is actually bigger; this means that there are no events between t_range[-1] and t_range[0]
            if t_range[-1] > events_handle.current_time:
                events_handle.seek_future_time(t_range[-1])
            else:
                events_handle.seek_past_time(t_range[-1])
            p_end = events_handle.current_event
            event_start_stops.append([p_start, p_end - p_start])
            file_idx_last = file_idx
        return np.array(event_start_stops, dtype=np.int64)

    def save_indices(self):
        savepath = self.get_event_indices_path()
        if savepath.exists():
            savepath.unlink()
        with h5py.File(savepath, "w") as hd:
            hd.create_dataset(f"event_indices", data=self.event_indices, dtype=np.int64)

    def load_indices(self):
        path = self.get_event_indices_path()
        with h5py.File(path, "r") as hd:
            event_indices = hd["event_indices"][:].astype(np.int64)
        return event_indices

    def get_box_times(self):
        sorted_paths = self.indexer.paths
        box_times = {}
        for idx_file, pp in enumerate(sorted_paths):
            name = Path(pp).name
            boxes_handle = PSEELoader(box_path_from_events_path(pp, bbox_suffix=self.bbox_suffix))
            boxes = load_boxes(boxes_handle)
            box_times[name] = boxes["t"]
        return box_times

    def get_path_from_idx(self, file_idx):
        return Path(self.indexer.paths[file_idx])

    def calc_start_times_old(self, box_times):
        t_start = box_times[0] - self.delta_t_mus
        t_end = box_times[-1]
        t_starts = np.arange(t_start, t_end + self.delta_t_mus, self.delta_t_mus)[:: self.n_seqs]
        return t_starts

    def calc_start_times(self, box_times):
        """Calculate start times of events and bboxs; 'Syncs' for each sequence, ie uses the closest time of
            bounding boxes as start time if possible.

        Args:
            box_times ([type]): [description]
            file_idx ([type]): [description]
        """
        start_times = []
        times_un = np.unique(box_times)
        n_steps = int((times_un[-1] - (times_un[0] - self.delta_t_mus)) / self.delta_t_mus)
        t_start = times_un[0] - self.delta_t_mus
        t_end = t_start + self.n_seqs * self.delta_t_mus  # t_range[-1]
        start_times.append(t_start)
        for ii in range(int(n_steps // self.n_seqs) - (1 * (n_steps % self.n_seqs == 0))):
            t_abs_diff = np.abs(times_un - t_end)
            closest_t_idx = np.argmin(t_abs_diff)
            if t_abs_diff[closest_t_idx] < self.dt_var_test_times_mus:
                t_start = times_un[closest_t_idx]
            else:
                t_start = t_end
            t_end = t_start + self.n_seqs * self.delta_t_mus  # t_range[-1]
            start_times.append(t_start)
        # can happen due to boundary conditions that we need one more sample; this is what we measure here
        if t_end + self.dt_var_bboxs_mus < times_un[-1]:
            t_abs_diff = np.abs(times_un - t_end)
            closest_t_idx = np.argmin(t_abs_diff)
            if t_abs_diff[closest_t_idx] < self.dt_var_test_times_mus:
                t_start = times_un[closest_t_idx]
            else:
                t_start = t_end
            start_times.append(t_start)
        start_times = np.array(start_times)
        return start_times

    def get_idx_to_seq_idx_map(self):
        # 0 -> 0; 1 -> n_seqs; 2 -> 2*n_seqs
        idx_to_seq_idx_map = []
        for file_idx, idx_max in enumerate(self.indexer.dset_idx_max_to_file_idx):
            box_times = self.box_times[self.get_path_from_idx(file_idx).name]
            if len(box_times) > 0:
                t_starts = self.calc_start_times(box_times)
                idx_to_seq_idx_map.append(np.c_[[file_idx] * len(t_starts), t_starts])
        return np.concatenate(idx_to_seq_idx_map)

    def get_file_idx(self, idx):
        if self.tvt == "train":
            file_idx = np.searchsorted(self.indexer.dset_idx_max_to_file_idx, idx)
            t_start = None
        else:
            # map idx such that only get each element of sequence once
            file_idx, t_start = self.idx_to_seq_idx_map[idx]
            file_idx = int(file_idx)
        return file_idx, t_start

    def get_t_range(self, idx):
        file_idx, t_start = self.get_file_idx(idx)
        if self.tvt == "train":
            time_idx = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 0], idx)
            if time_idx < self.n_seqs - 1:
                idx += self.n_seqs - 1
                time_idx = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 0], idx)
            t_end = self.indexer.file_idx_to_dset_idx[file_idx][time_idx, 1]
            t_start = t_end - self.n_seqs * self.delta_t_mus
        else:
            t_end = t_start + self.n_seqs * self.delta_t_mus
        t_range = np.linspace(t_start, t_end, num=self.n_seqs + 1, endpoint=True)
        t_range[t_range < 0] = 0.0
        return file_idx, t_range

    def call_train(self, idx):
        """Do backwards lookup: idx gives 'current' time t; time range is [t-n*dt, t]
        Doing it like this ensures that we have at least bounding boxes at the last time step;
        Otherwise, could happen that we load a sample where only the first step has bounding boxes,
        but is not included in the loss, leading to a de-facto empty sample
        """
        file_idx, t_range = self.get_t_range(idx)
        if self.pre_index_events:
            event_start_count = self.event_indices[idx]
        else:
            event_start_count = None
        bbox_start_ends = []
        box_times = self.box_times[self.get_path_from_idx(file_idx).name]
        for ii in range(self.n_seqs):
            t_end_this = t_range[self.n_seqs - ii] + self.dt_var_bboxs_mus
            t_start_this = t_end_this - 2 * self.dt_var_bboxs_mus
            bbox_start_end = np.searchsorted(box_times, [t_end_this, t_start_this])
            bbox_start_ends.append(bbox_start_end)
        return (
            file_idx,
            t_range,
            event_start_count,
            np.stack(bbox_start_ends)[::-1, ::-1],
        )

    def call_test(self, idx):
        """Do forward lookup: idx gives 'current' time t; time range is [t, t+n*dt]
        This is simpler because we want to process the whole sequence from start to end, even if there are
        no bounding boxes for some part of the sequence
        """
        file_idx, t_range = self.get_t_range(idx)
        if self.pre_index_events:
            event_start_count = self.event_indices[idx]
        else:
            event_start_count = None
        bbox_start_ends = []
        box_times = self.box_times[Path(self.indexer.paths[file_idx]).name]
        for ii in range(self.n_seqs):
            t_start_this = t_range[ii + 1] - self.dt_var_bboxs_mus
            t_end_this = t_start_this + 2 * self.dt_var_bboxs_mus
            bbox_start_end = np.searchsorted(box_times, [t_start_this, t_end_this])
            bbox_start_ends.append(bbox_start_end)
        return file_idx, t_range, event_start_count, np.stack(bbox_start_ends)

    def __call__(self, idx):
        if self.tvt == "train":
            return self.call_train(idx)
        else:
            return self.call_test(idx)

    def __len__(self):
        return self.len


class Proph1MpxOD(torch.utils.data.Dataset):
    def __init__(self, args_dict, train_val_test, pre_index_events=True, bbox_suffix="filtered"):
        self.args_dict = args_dict
        self.test_is_train = args_dict["test_dset_is_train_dset"]
        self.tvt = train_val_test
        self.bbox_suffix = bbox_suffix
        self.ns = args_dict["shape_t"][0]
        self.return_frames = args_dict["random_move_mnist_frames"]
        self.attrs = get_dataset_attributes(args_dict["dataset"])
        self.n_bins = args_dict["n_bins"]
        tvt = "train" if self.test_is_train else self.tvt
        self.root_path = proph_1mpx_paths[tvt]
        self.shape = self.attrs["shape_max"]
        self.indexer = Prophesee1MpxSeqIndexer(self.ns, tvt, pre_index_events=pre_index_events, bbox_suffix=bbox_suffix)
        self.events_transform = self.build_events_transform()
        self.seq_transform = self.build_seq_transform()
        self.bbox_transform = self.build_bbox_transform()
        self.bboxs = None
        self.labels = None
        self.bbox_times_ms = None
        self.event_handles = {}

    def load_events_by_time(self, file_idx, t_start, t_stop):
        path = self.indexer.get_path_from_idx(file_idx)
        events_handle = PSEELoader(str(path))
        events_handle.seek_time(t_stop)
        p_end = events_handle.current_event
        events_handle.seek_past_time(t_start)
        p_start = events_handle.current_event
        events = events_handle.load_n_events(p_end - p_start)
        return proph_1mpx_events_to_events(events, shift_time_to_zero=False)

    def load_events(self, file_idx, start_idx, n_events):
        path = self.indexer.get_path_from_idx(file_idx)
        if not file_idx in list(self.event_handles.keys()):
            self.event_handles[file_idx] = PSEELoader(str(path))
        events_handle = self.event_handles[file_idx]
        events_handle.seek_event(start_idx)
        events = events_handle.load_n_events(n_events)
        return proph_1mpx_events_to_events(events, shift_time_to_zero=False)

    def load_labels(self, idx):
        file_idx, t_range_mus, event_start_count, bbox_start_ends = self.indexer(idx)
        # convert to int here, because if contains NaN => has to be of type float
        start = bbox_start_ends[0, 0]
        stop = bbox_start_ends[-1, 1]
        boxes_handle = PSEELoader(
            box_path_from_events_path(self.indexer.get_path_from_idx(file_idx), bbox_suffix=self.bbox_suffix)
        )
        boxes_handle.seek_event(start)
        boxes = boxes_handle.load_n_events(stop - start)
        bboxs, labels = proph_1mpx_boxes_to_array(boxes)
        n_boxes = bbox_start_ends[:, 1] - bbox_start_ends[:, 0]
        return (
            file_idx,
            bboxs.astype(np.float32),
            labels.astype(int),
            t_range_mus,
            event_start_count,
            n_boxes,
        )

    def build_events_transform(self):
        return identity_transform

    def build_seq_transform(self):
        shape = self.args_dict["shape_t"][-2:]
        transform = VoxelGridTransform(shape)
        return transform

    def build_bbox_transform(self):
        # here, explicitly want to use self.tvt, even if test_is_train is True!
        if self.tvt == "train":
            transform = BBoxTransformRandomMoveMnist(self.args_dict)
        else:
            transform = BBoxTransformValTestRandomMoveMnist(self.args_dict["shape_t"][-2:])
        return transform

    def index_transform(self, idx, n_labels, n_boxes_per_seq):
        idxs = torch.full((n_labels, 2), -1000, dtype=torch.long)
        idxs[:, 0] = idx
        idxs_seq = np.insert(np.cumsum(n_boxes_per_seq), 0, 0)
        # during validation, pad sequence to be able to load as batch
        len_sample = len(idxs_seq) - 1 - np.isnan(idxs_seq).sum()
        idxs_seq = idxs_seq[: len_sample + 1].astype(int)
        for ii in range(len(idxs_seq) - 1):
            idxs[idxs_seq[ii] : idxs_seq[ii + 1], 1] = ii
        return idxs

    def __getitem__(self, idx):
        (
            file_idx,
            bboxs,
            labels,
            time_range_mus,
            event_start_count,
            n_boxes_per_seq,
        ) = self.load_labels(idx)
        if event_start_count is None:
            events = self.load_events_by_time(file_idx, time_range_mus[0], time_range_mus[-1])
        else:
            events = self.load_events(file_idx, event_start_count[0], event_start_count[1])
        events, bboxs, labels = self.events_transform(events, bboxs, labels)
        seq = events_to_seq_voxel_grid_2c(events, time_range_mus, self.n_bins, self.shape[1], self.shape[0])
        seq, bboxs, labels = self.seq_transform(seq, bboxs, labels)
        bboxs, labels = self.bbox_transform(bboxs, labels, n_boxes_per_seq=n_boxes_per_seq)
        idxs = self.index_transform(idx, len(labels), n_boxes_per_seq)
        frame_times_ms = torch.from_numpy(time_range_mus / 1e3)
        return (
            seq,
            bboxs,
            labels,
            idxs,
            torch.tensor([idx]),
            torch.tensor([file_idx]),
            frame_times_ms,
        )

    def __len__(self):
        return len(self.indexer)


class BBoxTransformRandomMoveMnist:
    def __init__(self, args_dict):
        self.args_dict = args_dict
        center_form_priors = PriorBox(
            args_dict["shape_t"],
            args_dict["prior_feature_maps"],
            args_dict["prior_min_sizes"],
            args_dict["prior_max_sizes"],
            args_dict["prior_strides"],
            args_dict["prior_aspect_ratios"],
            args_dict["prior_clip"],
            debug_mode=False,
        )()
        self.transform = SSDTargetTransform(
            center_form_priors,
            args_dict["shape_t"][-2:],
            args_dict["prior_center_variance"],
            args_dict["prior_size_variance"],
            args_dict["train_iou_threshold"],
            only_best_priors=False,
            debug_mode_priors=False,
            boxes_to_locations=args_dict["boxes_to_locations"],
            iou_func=args_dict["train_iou_func"],
        )

    def __call__(self, bboxs, labels, n_boxes_per_seq):
        # background is defined as 0
        labels += 1
        # transform bboxs separately for each seq
        idxs = np.insert(np.cumsum(n_boxes_per_seq), 0, 0)
        bboxs_trans = []
        labels_trans = []
        for ii in range(len(idxs) - 1):
            bt = bboxs[idxs[ii] : idxs[ii + 1]]
            lt = labels[idxs[ii] : idxs[ii + 1]]
            b_trans, l_trans = self.transform(bt, lt)
            bboxs_trans.append(b_trans)
            labels_trans.append(l_trans)
        return np.stack(bboxs_trans), np.stack(labels_trans)


class BBoxBackTransformRandomMoveMnist:
    """
    WARN: This is not actually used, look at SSDBoxHead and PostProcessor for code
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict
        center_form_priors = PriorBox(
            args_dict["shape_t"],
            args_dict["prior_feature_maps"],
            args_dict["prior_min_sizes"],
            args_dict["prior_max_sizes"],
            args_dict["prior_strides"],
            args_dict["prior_aspect_ratios"],
            args_dict["prior_clip"],
            debug_mode=False,
        )()
        self.transform = SSDTargetBackTransform(
            center_form_priors,
            args_dict["shape_t"][-2:],
            args_dict["prior_center_variance"],
            args_dict["prior_size_variance"],
            args_dict["train_iou_threshold"],
            only_best_priors=False,
            debug_mode_priors=False,
        )

    def __call__(self, locations, labels):
        b_trans, l_trans = self.transform(locations, labels)
        return b_trans, l_trans


class BBoxTransformValTestRandomMoveMnist:
    def __init__(self, shape):
        # (h, w)
        self.shape = shape

    def __call__(self, bboxs, labels, n_boxes_per_seq=None):
        bboxs = torch.from_numpy(bboxs)
        labels = torch.from_numpy(labels)
        # background is defined as 0
        labels += 1
        bboxs = normalize_boxes(bboxs, self.shape, copy=False)
        bboxs = box_convert(bboxs, "xywh", "xyxy")
        return bboxs, labels
