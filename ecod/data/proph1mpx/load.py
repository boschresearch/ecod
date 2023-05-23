# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import numpy as np

import torch

from ecod.events.events import events_to_frames
from ecod.data.proph1mpx.toolbox.psee_loader import PSEELoader
from ecod.paths import proph_1mpx_aux_data_path, proph_1mpx_paths
from ecod.utils.general import ProgressLogger
from ecod.utils.files import makedirs
from ecod.data.augmentations import crop_events_center, get_shape_from_events


def proph_1mpx_boxes_to_array(boxes):
    bboxs = np.c_[boxes["x"], boxes["y"], boxes["w"], boxes["h"]]
    labels = np.array(boxes["class_id"])
    return bboxs, labels


def load_boxes(boxes_handle):
    return boxes_handle.load_n_events(boxes_handle.event_count())


def load_boxes_by_path(path, bbox_suffix="filtered"):
    if str(path).endswith(".dat"):
        path = box_path_from_events_path(path, bbox_suffix=bbox_suffix)
    boxes_handle = PSEELoader(str(path))
    return boxes_handle.load_n_events(boxes_handle.event_count())


def box_path_from_events_path(filepath, bbox_suffix="filtered", raise_error=True):
    # somehow, some are named bbox.npy and some are named box.npy, check for both
    filepath = Path(filepath)
    for box_or_bbox in ["box", "bbox"]:
        box_path = Path(str(filepath).replace("_td.dat", f"_{box_or_bbox}.npy"))
        if box_path.exists():
            break
    if not box_path.exists():
        raise ValueError(
            f"The original (unfiltered) ground truth bounding boxes of file {filepath} are missing: " f"No {box_path}."
        )
    # now that we know that the unfiltered path exists, check for the filtered path
    elif bbox_suffix == "nofilter":
        return box_path
    elif bbox_suffix == "filtered" or bbox_suffix.startswith("only_moving"):
        box_path = Path(str(box_path).replace(f"_{box_or_bbox}.npy", f"_{box_or_bbox}_{bbox_suffix}.npy"))
    else:
        raise ValueError(f"bbox_suffix has to be 'nofilter', 'filtered', or 'only_movingXX' but is: {bbox_suffix}")
    if box_path.exists() or not raise_error:
        return box_path
    raise ValueError(f"The (filtered) ground truth bounding boxes of file {filepath} are missing: No {box_path}.")


def get_all_data_positions(which="val", delta_t=20000, print_progress=True, bbox_suffix="filtered"):
    paths = [str(aa) for aa in proph_1mpx_paths[which].glob("*.dat")]
    box_positions = {}
    box_times = {}
    events_positions = {Path(pp).name: [] for pp in paths}
    if print_progress:
        it = ProgressLogger(paths)
    else:
        it = paths
    for ii, pp in enumerate(it):
        name = Path(pp).name
        events_handle = PSEELoader(str(Path(pp)))
        boxes_handle = PSEELoader(box_path_from_events_path(pp, bbox_suffix=bbox_suffix))
        boxes = load_boxes(boxes_handle)
        times = np.unique(boxes["t"])
        times = times[times > delta_t]
        time_positions = np.searchsorted(boxes["t"], times)
        box_times[name] = times
        time_pos_pairs = np.concatenate([time_positions, [len(boxes["t"])]])
        time_pos_pairs = np.roll(np.repeat(time_pos_pairs, 2), -1)[:-2].reshape(-1, 2)
        box_positions[name] = time_pos_pairs
        for time in times:
            events_handle.seek_time(time)
            p_end = events_handle.current_event
            events_handle.seek_past_time(time - delta_t)
            p_start = events_handle.current_event
            events_positions[name].append((p_start, p_end - p_start))
        events_positions[name] = np.array(events_positions[name])
    return events_positions, box_positions, box_times


def extract_positions(which, delta_t):
    events_positions, box_positions, box_times = get_all_data_positions(which, delta_t)
    makedirs(proph_1mpx_aux_data_path, overwrite=False)
    names = ["events_positions", "box_positions", "box_times"]
    for name, arrs in zip(names, [events_positions, box_positions, box_times]):
        savepath = proph_1mpx_aux_data_path / f"{name}_{which}_dt{delta_t}.npz"
        if savepath.exists():
            savepath.unlink()
        np.savez(savepath, **arrs)


def proph_1mpx_events_to_events(events, shift_time_to_zero=True):
    if shift_time_to_zero:
        time = events["t"] - events["t"][0]
    else:
        time = events["t"]
    return np.c_[time, events["x"], events["y"], events["p"]].astype(np.float32)


class SimplePropheseeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        which,
        shape_t,
        transform=None,
        timestep=1,
        binary_spikes=True,
        seed=51252,
        events_transform=None,
        bbox_suffix="filtered",
    ):
        self.which = which
        self.shape_t = shape_t
        self.transform = transform
        self.events_transform = events_transform
        self.timestep = timestep
        self.binary_spikes = binary_spikes
        self.n_timesteps = self.shape_t[0]
        self.delta_t_mus = self.n_timesteps * self.timestep * 1e3
        self.rng = np.random.default_rng(seed=seed)
        self.paths = sorted([str(pp) for pp in proph_1mpx_paths[which].glob("*.dat")])
        self.box_paths = [box_path_from_events_path(pp, bbox_suffix=bbox_suffix) for pp in self.paths]
        self.box_times = None
        self.idxs = None

    def load_sample(self, idx):
        if self.box_times is None:
            self.box_times, self.idxs = self.load_box_times()
        path_idx, time_idx = self.idxs[idx]
        events_handle = PSEELoader(self.paths[path_idx])
        boxes_handle = PSEELoader(self.box_paths[path_idx])
        time = self.box_times[path_idx][time_idx]
        events_handle.seek_time(time - self.delta_t_mus)
        events = events_handle.load_delta_t(self.delta_t_mus)
        events = proph_1mpx_events_to_events(events)
        boxes_handle.seek_time(time)
        # want to have boxes at exactly this time (+ 10 microseconds just to be sure)
        boxes = boxes_handle.load_delta_t(10)
        boxes, labels = proph_1mpx_boxes_to_array(boxes)
        return events, boxes, labels

    def transform_events(self, events):
        shape = get_shape_from_events(events)
        if self.events_transform:
            events = self.events_transform(events, self.shape_t[2:])
        else:
            # events = trim_events(events, self.shape_t[2:])
            events = crop_events_center(events, self.shape_t[2:])
        return events, shape

    def load_box_times(self):
        box_times = []
        idxs = []
        for path_idx, path in enumerate(self.box_paths):
            boxes_handle = PSEELoader(path)
            boxes = load_boxes(boxes_handle)
            times = boxes["t"]
            # remove times smaller delta_t_mus, because don't have enough events for them
            times = times[times > self.delta_t_mus]
            box_times.append(times)
            idxs.append(np.c_[[path_idx] * len(times), np.arange(len(times))])
        return box_times, np.concatenate(idxs)

    def event_frame_sequence_from_events(self, events):
        frames = events_to_frames(
            events,
            self.shape_t[2:],
            n_frames=self.n_timesteps,
            t_start=-1,
            t_end=-1,
            binary_spikes=self.binary_spikes,
        )
        return frames

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        events, boxes, labels = self.load_sample(index)
        events, shape = self.transform_events(events)
        shape = torch.from_numpy(shape).long()
        event_frame_seq = self.event_frame_sequence_from_events(events)
        if self.transform:
            event_frame_seq = self.transform(event_frame_seq)
        else:
            event_frame_seq = torch.from_numpy(event_frame_seq).float()
            # from torchvision.transforms.transforms import GaussianBlur
            # self.gauss_blur = GaussianBlur(5)
            # event_frame_seq = self.gauss_blur(event_frame_seq)
        # not needed anymore, padding is already done in event_frame_sequence_from_events
        # event_frame_seq = pad_array(event_frame_seq, self.shape_t[-2:])
        return event_frame_seq, boxes, labels, shape, self.idxs[index]


def trim_boxes(boxes, shape):
    boxes = boxes.copy()
    x2s = boxes["x"] + boxes["w"]
    y2s = boxes["y"] + boxes["h"]
    boxes["x"][boxes["x"] < 0] = 0
    boxes["y"][boxes["y"] < 0] = 0
    x2s[x2s < 0] = 0
    y2s[y2s < 0] = 0
    boxes[boxes["x"] > shape[1] - 1] = shape[1] - 1
    x2s[x2s > shape[1] - 1] = shape[1] - 1
    boxes[boxes["y"] > shape[0] - 1] = shape[0] - 1
    y2s[y2s > shape[0] - 1] = shape[0] - 1
    boxes["w"] = x2s - boxes["x"]
    boxes["h"] = y2s - boxes["y"]
    boxes["w"][boxes["w"] < 0] = 0
    boxes["h"][boxes["h"] < 0] = 0
    return boxes
