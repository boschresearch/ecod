# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import numpy as np
import h5py

import torch
from torchvision.ops import box_convert

from ecod.utils.data import get_dataset_attributes, add_fake_bins_and_chans
from ecod.utils.files import load_h5_by_keys, load_json
from ecod.utils.general import ProgressLogger
from ecod.paths import random_move_mnist36_root, random_move_debug_root
from ecod.data.sim.rmmnist import load_rmmnist_events
from ecod.utils.data import identity_transform
from ecod.events.voxel import events_to_seq_voxel_grid_2c
from ecod.data.box2d.priors import PriorBox
from ecod.data.transforms import (
    SSDTargetTransform,
    SSDTargetBackTransform,
    VoxelGridTransform,
)
from ecod.data.box2d.transforms import normalize_boxes
from ecod.data.voxel.transforms import (
    RandomSampleCrop,
    Compose,
    RandomHorizontalMirror,
)


def load_bboxs(path):
    return load_h5_by_keys(path, ["bboxs_tlxywh"])["bboxs_tlxywh"]


def filter_events(events, t_start_mus, t_stop_mus, x_start, x_stop, y_start, y_stop):
    events = events.copy()
    for idx, val in zip([0, 1, 2], [t_start_mus, x_start, y_start]):
        mask = events[:, idx] > val
        events = events[mask]
    for idx, val in zip([0, 1, 2], [t_stop_mus, x_stop, y_stop]):
        mask = events[:, idx] < val
        events = events[mask]
    return events


def file_idx_from_path(path):
    return int(str(Path(path.name).with_suffix("")))


def count_events_in_bboxs(bbox_path, delta_t_ms):
    bbox_path = Path(bbox_path)
    root = bbox_path.parents[1]
    tvt = bbox_path.parent.name
    file_idx = file_idx_from_path(bbox_path)
    events = load_rmmnist_events(root, file_idx, tvt, to_float=True)
    bboxs = load_bboxs(bbox_path)
    data = load_h5_by_keys(bbox_path, ["bboxs_tlxywh", "labels", "bbox_times_ms"])
    bboxs, labels, bbox_times_ms = (
        data["bboxs_tlxywh"],
        data["labels"],
        data["bbox_times_ms"],
    )
    event_counts = []
    for bbox, label, time_ms in zip(bboxs, labels, bbox_times_ms):
        x_stop = bbox[0] + bbox[2]
        y_stop = bbox[1] + bbox[3]
        t_start_mus = (time_ms - delta_t_ms) * 1e3
        events_filtered = filter_events(events, t_start_mus, time_ms * 1e3, bbox[0], x_stop, bbox[1], y_stop)
        event_counts.append(len(events_filtered))
    return np.array(event_counts)


class RandomMoveMnistSingleIndexer:
    def __init__(self, root, bbox_suffix, train_val_test="train"):
        self.root = Path(root)
        meta_info_path = self.root / "meta_info.json"
        self.meta_data = load_json(meta_info_path)
        self.delta_t_ms = 1000.0 / self.meta_data["frames.fps"]
        self.bbox_suffix = bbox_suffix
        self.tvt = train_val_test
        if self.get_indices_path().exists():
            (
                self.file_idx_to_dset_idx,
                self.dset_idx_max_to_file_idx,
            ) = self.load_indices()
        else:
            self.file_idx_to_dset_idx, self.dset_idx_max_to_file_idx = self.index_dset()
            self.save_indices()
        self.len = max(self.dset_idx_max_to_file_idx)

    def get_sorted_paths(self):
        path = self.root / self.tvt
        return sorted(
            [pp for pp in path.glob("*.h5") if not (pp.name.startswith("index_map") or "metadata" in pp.name)]
        )

    def index_dset(self):
        file_idx_to_dset_idx = {}
        dset_idx_max_to_file_idx = []
        sorted_paths = self.get_sorted_paths()
        idx_glob = 0
        for pp in ProgressLogger(sorted_paths, name="INDEXER"):
            data = self.get_filtered_label_info(pp, idx_glob)
            idx_file = int(pp.with_suffix("").name)
            file_idx_to_dset_idx[idx_file] = data
            idx_glob += len(data)
            dset_idx_max_to_file_idx.append(idx_glob - 1)
        return file_idx_to_dset_idx, dset_idx_max_to_file_idx

    def get_filter_mask(self, path):
        if self.bbox_suffix == "none":
            return None
        elif self.bbox_suffix == "only_moving_diff":
            with h5py.File(path, "r") as hd:
                bboxs = hd["bboxs_tlxywh"][:].astype(float)
                # moving object at t is defined as box[t] != box[t-1]
                # WARN: This does only work for one object; for multiple objects, would need tracking ID or speed
                moving_mask = np.concatenate([[False], (np.diff(bboxs, axis=0) > 0).any(1)])
                return moving_mask
        elif self.bbox_suffix == "only_moving":
            count_path = path.parent / f"{Path(path.name).with_suffix('')}_metadata.h5"
            if count_path.exists():
                with h5py.File(count_path, "r") as hd:
                    event_counts = hd["event_counts"][:]
            else:
                event_counts = count_events_in_bboxs(path, delta_t_ms=self.delta_t_ms)
                with h5py.File(count_path, "w") as hd:
                    hd.create_dataset("event_counts", data=event_counts)
            thresh = 0.0
            mask = event_counts > thresh
            return mask
        else:
            raise ValueError(
                f"bbox_suffix has to be in ['none', 'only_moving', 'only_moving_diff'], but is {self.bbox_suffix}"
            )

    def get_filtered_label_info(self, path, idx_glob):
        with h5py.File(path, "r") as hd:
            times_ms = hd["frame_times_ms"][:]
            bbox_times_ms = hd["bbox_times_ms"][:]
            bbox_idxs = hd["bbox_idxs"][:]
        frame_idxs = np.arange(len(times_ms))
        bbox_starts = np.searchsorted(bbox_idxs, frame_idxs)
        bbox_ends = bbox_starts.copy()[1:]
        bbox_ends = np.append(bbox_ends, np.searchsorted(bbox_idxs, len(times_ms)))
        data = np.c_[bbox_starts, bbox_ends, bbox_times_ms]
        mask = self.get_filter_mask(path)
        if mask is not None:
            data = data[mask]
            frame_idxs = frame_idxs[mask]
        data = np.c_[np.arange(len(data)) + idx_glob, frame_idxs, data]
        return data

    def get_indices_path(self):
        suf = f"_{self.bbox_suffix}" if self.bbox_suffix != "none" else ""
        return self.root / self.tvt / f"index_map{suf}.h5"

    def save_indices(self):
        savepath = self.get_indices_path()
        if savepath.exists():
            savepath.unlink()
        with h5py.File(savepath, "w") as hd:
            for idx, arr in self.file_idx_to_dset_idx.items():
                hd.create_dataset(f"{idx:0>5}", data=arr, dtype=np.float64)
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
        internal_idx = np.searchsorted(self.file_idx_to_dset_idx[file_idx][:, 0], idx)
        frame_idx = int(self.file_idx_to_dset_idx[file_idx][internal_idx, 1])
        bbox_start = int(self.file_idx_to_dset_idx[file_idx][internal_idx, 2])
        bbox_end = int(self.file_idx_to_dset_idx[file_idx][internal_idx, 3])
        return file_idx, frame_idx, bbox_start, bbox_end

    def __len__(self):
        return self.len


class RandomMoveMnistSeqIndexer:
    """
    When drawing multiple bboxs, have to make sure that we don't draw at the end,
    because then the sequence is not long enough. We 'cheat' here and just return another index.
    This is okay during training, but should not be used during val or test
    """

    def __init__(self, root, n_seqs, bbox_suffix=None, train_val_test="train"):
        self.root = root
        self.n_seqs = n_seqs
        self.bbox_suffix = bbox_suffix
        self.tvt = train_val_test
        self.indexer = RandomMoveMnistSingleIndexer(root, bbox_suffix, self.tvt)
        self.delta_t_ms = self.indexer.delta_t_ms
        self.dt_var_bboxs_ms = 0.49 * self.delta_t_ms
        self.dt_var_test_times_ms = 0.2 * self.delta_t_ms
        self.box_times, self.frame_times = self.get_box_times()
        self.idx_to_seq_idx_map = self.get_idx_to_seq_idx_map()
        self.len = len(self.indexer) if self.tvt == "train" else len(self.idx_to_seq_idx_map)

    def get_box_times(self):
        sorted_paths = self.indexer.get_sorted_paths()
        box_times = {}
        frame_times = {}
        for idx_file, pp in enumerate(sorted_paths):
            with h5py.File(pp, "r") as hd:
                times = hd["bbox_times_ms"][:]
                f_times = hd["frame_times_ms"][:]
            box_times[idx_file] = times
            frame_times[idx_file] = f_times
        return box_times, frame_times

    def calc_start_times(self, box_times):
        """Calculate start times of events and bboxs; 'Syncs' for each sequence, ie uses the closest time of
            bounding boxes as start time if possible.

        Args:
            box_times ([type]): [description]
            file_idx ([type]): [description]
        """
        start_times = []
        times_un = np.unique(box_times)
        if len(times_un) == 0:
            return np.array(start_times)
        n_steps = int((times_un[-1] - (times_un[0] - self.delta_t_ms)) / self.delta_t_ms)
        t_start = times_un[0] - self.delta_t_ms
        t_end = t_start + self.n_seqs * self.delta_t_ms  # t_range[-1]
        start_times.append(t_start)
        for ii in range(int(n_steps // self.n_seqs) - (1 * (n_steps % self.n_seqs == 0))):
            t_abs_diff = np.abs(times_un - t_end)
            closest_t_idx = np.argmin(t_abs_diff)
            if t_abs_diff[closest_t_idx] < self.dt_var_test_times_ms:
                t_start = times_un[closest_t_idx]
            else:
                t_start = t_end
            t_end = t_start + self.n_seqs * self.delta_t_ms  # t_range[-1]
            start_times.append(t_start)
        # can happen due to boundary conditions that we need one more sample; this is what we measure here
        if t_end + self.dt_var_bboxs_ms < times_un[-1]:
            t_abs_diff = np.abs(times_un - t_end)
            closest_t_idx = np.argmin(t_abs_diff)
            if t_abs_diff[closest_t_idx] < self.dt_var_test_times_ms:
                t_start = times_un[closest_t_idx]
            else:
                t_start = t_end
            start_times.append(t_start)
        start_times = np.array(start_times)
        return start_times

    def get_idx_to_seq_idx_map(self):
        # 0 -> 0; 1 -> n_seqs; 2 -> 2*n_seqs
        idx_to_seq_idx_map = []
        paths = self.indexer.get_sorted_paths()
        for file_idx, idx_max in enumerate(self.indexer.dset_idx_max_to_file_idx):
            box_times = self.box_times[file_idx]
            path = paths[file_idx]
            mask = self.indexer.get_filter_mask(path)
            if mask is not None:
                box_times = box_times[mask]
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
            internal_idx = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 0], idx)
            frame_idx = int(self.indexer.file_idx_to_dset_idx[file_idx][internal_idx, 1])
            if frame_idx < self.n_seqs - 1:
                idx += self.n_seqs - 1
                internal_idx = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 0], idx)
            t_end = self.indexer.file_idx_to_dset_idx[file_idx][internal_idx, 4]
            t_start = t_end - self.n_seqs * self.delta_t_ms
        else:
            t_end = t_start + self.n_seqs * self.delta_t_ms
        t_range = np.linspace(t_start, t_end, num=self.n_seqs + 1, endpoint=True)
        t_range[t_range < 0] = 0.0
        return file_idx, t_range, idx

    def call_train(self, idx):
        """
        bbox_starts: BE CAREFUL WHEN USING.
            Indices from frames_idx-1 until frames_idx+n_seqs to cover time before first and after last bounding box.
            For n_seqs=4, len(bbox_starts) == n_seqs+2
        """
        file_idx, t_range, idx = self.get_t_range(idx)
        internal_idx = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 0], idx)
        frame_stop = int(self.indexer.file_idx_to_dset_idx[file_idx][internal_idx, 1])
        frame_start = frame_stop - self.n_seqs + 1
        bbox_start_ends = [self.indexer.file_idx_to_dset_idx[file_idx][internal_idx, 2:4].astype(int)]
        n_boxes = [bbox_start_ends[0][1] - bbox_start_ends[0][0]]
        for ii in range(1, self.n_seqs):
            frame_idx_now = frame_stop - ii
            internal_idx_now = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 1], frame_idx_now)
            if internal_idx_now < len(self.indexer.file_idx_to_dset_idx[file_idx]) and frame_idx_now == int(
                self.indexer.file_idx_to_dset_idx[file_idx][internal_idx_now, 1]
            ):
                bbox_start_ends.append(self.indexer.file_idx_to_dset_idx[file_idx][internal_idx_now, 2:4].astype(int))
                n_boxes.append([bbox_start_ends[-1][1] - bbox_start_ends[-1][0]])
            else:
                bbox_start_ends.append([bbox_start_ends[-1][0], bbox_start_ends[-1][0]])
                n_boxes.append(0)
        bbox_start_ends = np.stack(bbox_start_ends)[::-1]
        bbox_start_ends = np.concatenate([bbox_start_ends[:, 0], [bbox_start_ends[-1, -1]]])
        return file_idx, [frame_start, frame_stop], t_range, bbox_start_ends

    def call_test(self, idx):
        """Do forward lookup: idx gives 'current' time t; time range is [t, t+n*dt]
        This is simpler because we want to process the whole sequence from start to end, even if there are
        no bounding boxes for some part of the sequence
        """
        file_idx, t_range, idx = self.get_t_range(idx)
        time_diffs = np.abs(self.frame_times[file_idx].reshape(-1, 1) - t_range[1:].reshape(1, -1))
        frame_idxs = np.argmin(time_diffs, 0)
        # frame_idxs = frame_idxs[frame_idxs < len(self.frame_times[file_idx])]
        n_seqs = frame_idxs[-1] + 1 - frame_idxs[0]
        frame_idxs = frame_idxs[: n_seqs + 1]
        bbox_start_ends = []
        for ii in range(n_seqs):
            frame_idx_now = frame_idxs[ii]
            internal_idx_now = np.searchsorted(self.indexer.file_idx_to_dset_idx[file_idx][:, 1], frame_idx_now)
            if internal_idx_now < len(self.indexer.file_idx_to_dset_idx[file_idx]) and frame_idx_now == int(
                self.indexer.file_idx_to_dset_idx[file_idx][internal_idx_now, 1]
            ):
                bbox_start_ends.append(self.indexer.file_idx_to_dset_idx[file_idx][internal_idx_now, 2:4].astype(int))
            else:
                bbox_start_ends.append([-1, -1])
        bbox_start_ends = np.stack(bbox_start_ends)
        n_bboxs = bbox_start_ends[:, 1] - bbox_start_ends[:, 0]
        start = bbox_start_ends[bbox_start_ends > -1]
        if len(start) > 0:
            start = start[0]
            bbox_start_ends = start + np.insert(np.cumsum(n_bboxs), 0, 0)
        else:
            bbox_start_ends = np.zeros((2,), dtype=int)
        return (
            file_idx,
            [frame_idxs[0], frame_idxs[-1]],
            t_range[: n_seqs + 1],
            bbox_start_ends,
        )

    def __call__(self, idx):
        if self.tvt == "train":
            return self.call_train(idx)
        else:
            return self.call_test(idx)

    def __len__(self):
        return self.len


class RandomMoveMnistOD(torch.utils.data.Dataset):
    def __init__(self, args_dict, train_val_test):
        self.args_dict = args_dict
        self.test_is_train = args_dict["test_dset_is_train_dset"]
        self.tvt = train_val_test
        self.ns = args_dict["shape_t"][0]
        self.return_frames = args_dict["random_move_mnist_frames"]
        self.attrs = get_dataset_attributes(args_dict["dataset"])
        self.root_path = self.get_root_path()
        meta_info_path = self.root_path / "meta_info.json"
        self.meta_data = load_json(meta_info_path)
        self.shape = self.meta_data["shape"]
        self.n_bins = args_dict["n_bins"]
        self.delta_t_seq_ms = 1000.0 / self.meta_data["frames.fps"]
        self.events_transform = self.build_events_transform()
        self.seq_transform = self.build_seq_transform()
        self.bbox_transform = self.build_bbox_transform()
        # load train indices if test_is_train
        tvt = "train" if self.test_is_train else self.tvt
        # even when test_is_train, want bbox suffix to be 'the right one'
        tvt_bboxs = "test" if self.tvt in ["val", "test"] else "train"
        self.indexer = RandomMoveMnistSeqIndexer(self.root_path, self.ns, args_dict[f"bbox_suffix_{tvt_bboxs}"], tvt)
        self.bboxs = None
        self.labels = None
        self.bbox_times_ms = None

    def get_root_path(self):
        if "debug" in self.args_dict["dataset"]:
            root_path = random_move_debug_root
        else:
            root_path = Path(random_move_mnist36_root)
        return root_path

    def load_events(self, file_idx):
        tvt = "train" if self.test_is_train else self.tvt
        events = load_rmmnist_events(self.root_path, file_idx, tvt, to_float=True)
        return events

    def load_events_h5_part(self, idx, t_start_ms, t_end_ms):
        tvt = "train" if self.test_is_train else self.tvt
        path = self.root_path / tvt / f"{idx:0>5}_events.h5"
        with h5py.File(path, "r") as hd:
            start, end = np.searchsorted(hd["events"]["t"], [t_start_ms * 1e6, t_end_ms * 1e6])
            events = hd["events"][start:end]
        return np.c_[events["t"] / 1e3, events["x"], events["y"], events["p"]]

    def load_sample(self, idx, with_frames=False):
        file_idx, frame_start_stop, frame_times_ms, bbox_starts = self.indexer(idx)
        n_boxes = np.diff(bbox_starts)
        # convert to int here, because if contains NaN => has to be of type float
        bbox_starts = bbox_starts.astype(int)
        tvt = "train" if self.test_is_train else self.tvt
        path = self.root_path / tvt / f"{file_idx:0>5}.h5"
        with h5py.File(path, "r") as hd:
            bboxs = hd["bboxs_tlxywh"][bbox_starts[0] : bbox_starts[-1]]
            labels = hd["labels"][bbox_starts[0] : bbox_starts[-1]]
            if with_frames:
                # frame_stop is the last frame index -> +1 to load it
                frames = hd["frames"][frame_start_stop[0] : frame_start_stop[1] + 1]
            else:
                frames = None
        return (
            file_idx,
            frames,
            bboxs.astype(np.float32),
            labels.astype(int),
            frame_times_ms,
            n_boxes,
        )

    def get_sample_from_preloaded_bboxs(self, idx):
        file_idx, frame_idx, bbox_starts = self.indexer(idx)
        # from 1 because want times *before* first bbox to cut out events
        bboxs = self.bboxs[file_idx][bbox_starts[1] : bbox_starts[-1]]
        labels = self.labels[file_idx][bbox_starts[1] : bbox_starts[-1]]
        bbox_times_ms = self.bbox_times_ms[file_idx][bbox_starts[0] : bbox_starts[-2]]
        return file_idx, bboxs.astype(np.float32), labels.astype(int), bbox_times_ms

    def build_events_transform(self):
        return identity_transform

    def build_seq_transform(self):
        shape = self.args_dict["shape_t"][-2:]
        resizer = VoxelGridTransform(shape)
        if self.tvt == "train":
            transform = []
            if self.args_dict["random_crop"]:
                random_cropper = RandomSampleCrop()
                transform.append(random_cropper)
            if self.args_dict["random_mirror"]:
                random_mirror = RandomHorizontalMirror()
                transform.append(random_mirror)
            transform = Compose(transform + [resizer])
        else:
            transform = resizer
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
            frames,
            bboxs,
            labels,
            frame_times_ms,
            n_boxes_per_seq,
        ) = self.load_sample(idx, with_frames=self.return_frames)
        if self.return_frames:
            seq = add_fake_bins_and_chans(torch.from_numpy(frames), self.n_bins, 2).numpy()
        else:
            events = self.load_events(file_idx)
            events, bboxs, labels = self.events_transform(events, bboxs, labels)
            seq = events_to_seq_voxel_grid_2c(events, frame_times_ms * 1e3, self.n_bins, self.shape[1], self.shape[0])
        seq, bboxs, labels = self.seq_transform(seq, bboxs, labels)
        bboxs, labels = self.bbox_transform(bboxs, labels, n_boxes_per_seq=n_boxes_per_seq)
        idxs = self.index_transform(idx, len(labels), n_boxes_per_seq)
        return (
            seq,
            bboxs,
            labels,
            idxs,
            torch.tensor([idx]),
            torch.tensor([file_idx]),
            torch.from_numpy(frame_times_ms),
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
            debug_mode_priors=args_dict["prior_assignment_debug_mode"],
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
            debug_mode_priors=args_dict["prior_assignment_debug_mode"],
            boxes_to_locations=self.args_dict["boxes_to_locations"],
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
