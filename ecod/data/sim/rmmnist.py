# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
from pathlib import Path
import numpy as np
import h5py
import json

from skimage.transform import resize

from ecod.utils.general import ProgressLogger
from ecod.utils.files import makedirs, load_json
from ecod.utils.data import get_dataset_attributes
from ecod.data.mnist.load import load_mnist_data

from ecod.data.sim.simulator import (
    get_config,
    simulate_events,
    events_np_to_float,
    process_frames,
    save_events,
)


def trim_paths(paths, shapes_max):
    """
    paths: (n_objects, 2, n_steps)
    """
    paths[paths < 0] = 0.0
    paths = np.swapaxes(paths, 0, 1)
    for ii, ss in enumerate(shapes_max):
        paths[0, ii, paths[0, ii] >= ss[1]] = ss[1] - 1
        paths[1, ii, paths[1, ii] >= ss[0]] = ss[0] - 1
    return np.swapaxes(paths, 0, 1)


class RandomMoveMnistGenerator:
    def __init__(
        self,
        time_s,
        shape,
        n_samples,
        delta_t_ms=1.0,
        n_objects_per_sample=1,
        labels_to_use=None,
        parameters_sim=None,
        train_val_test="train",
        split=0.9,
        seed=12,
    ):
        if seed == 0:
            raise ValueError("Don't use seed=0, because then all seeds will be the same.")
        self.time_s = time_s
        self.shape = shape
        self.delta_t_ms = delta_t_ms
        self.n_samples = n_samples
        self.n_objects_per_sample = n_objects_per_sample
        self.train_val_test = train_val_test
        self.split = split
        self.n_time_steps = int(time_s / delta_t_ms * 1000.0)
        self.labels_to_use = sorted(labels_to_use) if labels_to_use is not None else list(range(10))
        self.current_sample_idx = 0
        self.params = self.check_parameters_sim(parameters_sim)
        self.rng_locs = np.random.default_rng(seed=7 * seed)
        self.rng_size = np.random.default_rng(seed=11 * seed)
        self.rng_num = np.random.default_rng(seed=17 * seed)
        self.attrs = get_dataset_attributes("mnist")
        self.shape_mnist = self.attrs["shape_max"]
        self.min_size = int(max(min(self.shape) // 10, 20))
        self.max_size = int(max(min(shape) // 2, max(self.shape_mnist)))
        self.imgs, self.labels = self.load_data()
        self.idx_map = self.get_index_map()
        self.labels_samples = self.rng_num.choice(self.labels_to_use, (self.n_samples, n_objects_per_sample))
        self.check_labels(train_val_test)

    def check_labels(self, train_val_test):
        # need to only include labels that are actually going to be simulated for train;
        # for val, test,
        if train_val_test == "train":
            self.actual_labels = np.unique(self.labels_samples)
            self.label_name_map = {vv: ii for ii, vv in enumerate(sorted(self.actual_labels))}
        else:
            self.actual_labels = np.unique(self.labels_samples)
            if any([ll not in self.actual_labels for ll in self.labels_to_use]):
                raise RuntimeError(
                    f"Specified {train_val_test} set such that the following labels should be used: "
                    f"{self.labels_to_use}, while only the following labels are used: {self.actual_labels}. "
                    f"Consider increasing the number of samples or the number of objects per sample"
                )
            self.label_name_map = {vv: ii for ii, vv in enumerate(sorted(self.labels_to_use))}

    @staticmethod
    def get_default_params():
        params = dict(name="random_movements", move_rate_hz=0.5, move_time_ms=100.0)
        return params

    def check_parameters_sim(self, parameters):
        params = {}
        def_params = RandomMoveMnistGenerator.get_default_params()
        for kk, vv in def_params.items():
            params[kk] = parameters.get(kk, vv)
        params["n_steps_move"] = int(params["move_time_ms"] / self.delta_t_ms)
        return params

    def get_index_map(self):
        idx_map = {}
        idxs = np.arange(len(self.labels))
        for label_idx in self.labels_to_use:
            idx_map[label_idx] = idxs[self.labels == label_idx]
        return idx_map

    def load_data(self):
        return load_mnist_data(self.split, self.train_val_test, True)

    def locations_to_bounding_boxes(self, xs_tl, ys_tl):
        ss = (len(xs_tl),)
        bboxs = np.c_[
            xs_tl,
            ys_tl,
            np.full(ss, self.shape_mnist[0]),
            np.full(ss, self.shape_mnist[1]),
        ]
        return bboxs

    def get_tight_bbox_offset_from_image(self, img, th=5):
        if img.max() <= 1:
            th = th / 255
        shape = img.shape[:2]
        indices = np.mgrid[0 : shape[0], 0 : shape[1]].reshape(2, shape[0] * shape[1])
        mask = (img > th).reshape(-1)
        idxs_num = indices[:, mask]
        yx_tl = idxs_num.min(1)
        yx_br = idxs_num.max(1)
        hw = yx_br - yx_tl
        # xywh offsets to image location
        return np.concatenate([yx_tl[::-1], hw[::-1]])

    def draw_shapes_for_gen(self):
        shapes = self.rng_size.integers(self.min_size, self.max_size, (self.n_objects_per_sample, 2))
        return shapes

    def prepare_gen(self):
        idxs = []
        labels = self.labels_samples[self.current_sample_idx]
        for ii in range(self.n_objects_per_sample):
            load_idx = self.rng_num.choice(self.idx_map[labels[ii]], replace=False, size=None)
            idxs.append(load_idx)
        shapes = self.draw_shapes_for_gen()
        imgs = [resize(self.imgs[idx], shapes[ii]) for ii, idx in enumerate(idxs)]
        xywh_offsets = [self.get_tight_bbox_offset_from_image(img) for img in imgs]
        shapes_max = np.array(self.shape).reshape(1, 2) - shapes
        return imgs, labels, shapes, shapes_max, xywh_offsets

    def gen_sample(self):
        if self.current_sample_idx >= self.n_samples:
            raise ValueError(f"Sampled more than the given n_samples={self.n_samples}")
        imgs, labels, shapes, shapes_max, xywh_offsets = self.prepare_gen()
        paths = self.simulate_random_movements(shapes_max)
        xs_tl = paths[:, 0].T
        ys_tl = paths[:, 1].T

        seq = np.zeros((self.n_time_steps, 1, *self.shape), dtype=np.float32)
        bboxs_tlwh = []
        box_labels = []
        time_idxs = []
        for ii, (xx, yy) in enumerate(zip(xs_tl, ys_tl)):
            bboxs_sample = []
            labels_sample = []
            for jj, (img, x_n, y_n, ss) in enumerate(zip(imgs, xx, yy, shapes)):
                seq[ii, :, y_n : y_n + ss[0], x_n : x_n + ss[1]] = img
                offs = xywh_offsets[jj]
                bboxs_sample.append([x_n + offs[0], y_n + offs[1], offs[2], offs[3]])
                labels_sample.append(self.label_name_map[labels[jj]])
            bboxs_tlwh.append(bboxs_sample)
            box_labels.append(labels_sample)
            time_idxs.append(ii)
        # (t, n_objs, 4)
        bboxs_tlwh = np.array(bboxs_tlwh)
        box_labels = np.array(box_labels)
        time_idxs = np.array(time_idxs)
        # heights = np.tile(shapes[:, 0], self.n_time_steps)
        # widths = np.tile(shapes[:, 1], self.n_time_steps)
        # bboxs_tlwh = np.stack([xs_tl.ravel(), ys_tl.ravel(), widths, heights], 1)
        # labels = np.concatenate([[self.label_name_map[ll] for ll in labels] for _ in range(self.n_time_steps)], 0)
        # time_idxs = np.concatenate([[ii for _ in idxs] for ii in range(self.n_time_steps)], 0)
        speeds = self.get_speeds(xs_tl, ys_tl)
        stops = self.get_stops_as_connected_components(speeds)
        self.current_sample_idx += 1
        return seq, bboxs_tlwh, box_labels, time_idxs, speeds.T, stops.T

    def draw_random_movements_times(self, n_objects, delta_t_ms):
        # expected number of samples to cover time range
        n_samples_exp = max(int(self.params["move_rate_hz"] * self.time_s), 1)
        # to be sure, take 5 times the expected number
        move_times_s = self.rng_locs.exponential(
            1.0 / self.params["move_rate_hz"], size=(n_objects, 5 * n_samples_exp)
        ).cumsum(1)
        # always move at 0
        return np.concatenate([np.full((n_objects, 1), delta_t_ms / 1e3), move_times_s], 1)

    def draw_xy_next(self, shape_max, size):
        x_nexts = self.rng_locs.integers(shape_max[1], size=size)
        y_nexts = self.rng_locs.integers(shape_max[0], size=size)
        return x_nexts, y_nexts

    def draw_origin(self, shapes_max):
        dims = shapes_max.shape[1]
        origin = np.stack([self.rng_locs.integers(ss_max, size=(dims, 1)) for ss_max in shapes_max], 0)
        return origin

    def simulate_random_movements(self, shapes_max):
        """Move objects in random direction
        1. Draw move times (also: always move at 0)
            (exponential distribution gives time between two events, if event rate is poisson distributed)
        2. For each time, draw a random location
        4. Move to random location for move_time_ms milliseconds
        5. Stop until next move time
        """
        n_objects, dims = shapes_max.shape
        n_time_steps = int(self.time_s / self.delta_t_ms * 1000.0)
        n_steps_move = int(self.params["move_time_ms"] / self.delta_t_ms)
        steps = np.zeros((n_objects, dims, n_time_steps))
        origin = self.draw_origin(shapes_max)
        steps[:, :, :1] = origin[:, ::-1]
        move_times_s = self.draw_random_movements_times(n_objects, self.delta_t_ms)
        sim_times = np.arange(0, self.time_s * 1000.0, self.delta_t_ms)
        for sample_idx in range(n_objects):
            move_times_sample = move_times_s[sample_idx, move_times_s[sample_idx] < self.time_s]
            if len(move_times_sample) <= 0:
                continue
            # removed filtering close time indices. This leads to wrong calculations of locs. It only works because
            #  we trim the paths in the end.
            time_idxs_move = np.searchsorted(sim_times, move_times_sample * 1000.0)
            x_nexts, y_nexts = self.draw_xy_next(shapes_max[sample_idx], len(time_idxs_move))
            locs_move = np.stack([x_nexts, y_nexts])
            step_sizes = np.diff(np.concatenate([origin[sample_idx][::-1], locs_move], 1)) / n_steps_move
            for ii, t_idx in enumerate(time_idxs_move):
                steps[sample_idx, :, t_idx : t_idx + n_steps_move] = step_sizes[:, ii : ii + 1]
        paths = steps.cumsum(2)
        paths = trim_paths(paths, shapes_max)
        neg_coords = paths[paths.astype(int) <= -1]
        if len(neg_coords) > 0:
            raise RuntimeError(f"Found negative coords in paths: {neg_coords}, there has to be a bug somewhere")
        return paths.astype(int)

    def get_frame_idxs_and_times(self, fps):
        frame_idxs = np.linspace(
            0,
            self.time_s / self.delta_t_ms * 1000.0,
            num=int(self.time_s * fps),
            endpoint=False,
        )
        frame_times = (frame_idxs * self.delta_t_ms)[1:]
        return frame_idxs.astype(int)[1:], frame_times

    def get_speeds(self, xs_tl, ys_tl):
        speeds_x = np.concatenate([np.diff(xs_tl, axis=0) / self.delta_t_ms, np.zeros((1, xs_tl.shape[1]))], 0).T
        speeds_y = np.concatenate([np.diff(ys_tl, axis=0) / self.delta_t_ms, np.zeros((1, ys_tl.shape[1]))], 0).T
        speeds = np.stack([speeds_x, speeds_y], 0)
        return speeds

    @staticmethod
    def identify_connected_components(array_1d):
        cur_idx = np.ones((len(array_1d),))
        arr_out = np.zeros(array_1d.shape, dtype=int)
        mask = array_1d[:, 0] == 0
        arr_out[mask, 0] = cur_idx[mask]
        for ii, items in enumerate(array_1d[:, 1:].T):
            mask = items == 0
            arr_out[mask, ii + 1] = cur_idx[mask]
            mask = (items != 0) & (array_1d[:, ii] == 0)
            cur_idx[mask] += 1
        return arr_out

    @staticmethod
    def get_stops_as_connected_components(speeds):
        speeds_abs = np.sqrt((speeds**2).sum(0))
        con_comps = RandomMoveMnistGenerator.identify_connected_components(speeds_abs == 0)
        return con_comps


class DebugRandomMoveMnistGenerator(RandomMoveMnistGenerator):
    def __init__(
        self,
        time_s,
        shape,
        n_samples,
        delta_t_ms=1.0,
        n_objects_per_sample=1,
        labels_to_use=None,
        parameters_sim=None,
        train_val_test="train",
        split=0.9,
        seed=12,
    ):
        super().__init__(
            time_s,
            shape,
            n_samples,
            delta_t_ms,
            n_objects_per_sample,
            labels_to_use,
            parameters_sim,
            train_val_test,
            split,
            seed,
        )
        div, mod = divmod(n_samples, len(self.labels_to_use))
        if mod != 0:
            raise ValueError(f"n_samples has to be divisible by labels, but are {n_samples}, {labels_to_use}")
        # sample equally from each label
        self.labels_samples = np.repeat(self.labels_to_use, div).reshape(-1, 1)
        self.max_size = int(min(self.shape) * 0.95)
        self.all_shapes = np.tile([self.min_size, self.max_size // 2, self.max_size], self.n_samples).reshape(-1, 1)

    def load_data(self):
        imgs, labels = load_mnist_data(1.0, "train", True)
        imgs_tmp = []
        for label in self.labels_to_use:
            mask = labels == label
            imgs_tmp.append(imgs[mask][0])
        imgs, labels = np.stack(imgs_tmp, axis=0), np.array(self.labels_to_use)
        return imgs, labels

    def check_labels(self, train_val_test):
        self.actual_labels = self.labels_to_use
        self.label_name_map = {vv: ii for ii, vv in enumerate(sorted(self.actual_labels))}

    def draw_shapes_for_gen(self):
        shape = np.repeat(self.all_shapes[self.current_sample_idx], 2).reshape(1, 2)
        return shape

    def draw_random_movements_times(self, n_objects, delta_t_ms):
        return np.full((n_objects, 1), delta_t_ms / 1e3)

    def draw_xy_next(self, shape_max, size):
        x_nexts = np.full((size,), shape_max[1])
        y_nexts = np.full((size,), shape_max[0])
        return x_nexts, y_nexts

    def draw_origin(self, shapes_max):
        dims = shapes_max.shape[1]
        origin = np.stack([np.zeros((dims, 1), dtype=int) for _ in shapes_max], 0)
        return origin


def append_to_h5(h5_handle, name, data):
    h5_handle[name].resize(h5_handle[name].shape[0] + data.shape[0], axis=0)
    h5_handle[name][-data.shape :] = data


def create_dataset_folders(ad):
    nn = "rmov"
    logger = logging.getLogger("GEN")
    folder_name = "mov_mnist_{}_tms{}_dtmus{}_s0{}_s1{}_nm{}".format(
        nn,
        int(1000.0 * ad["time_s"]),
        int(1000 * ad["delta_t_ms"]),
        ad["shape"][0],
        ad["shape"][1],
        ad["n_objects_per_sample"],
    )
    if ad["debug"]:
        folder_name = f"debug_{folder_name}"
    if ad["labels"] is not None:
        folder_name += "_" + "-".join(str(ii) for ii in ad["labels"])
    root = Path(ad["savedir"]) / folder_name
    for train_val_test in ["train", "val", "test"]:
        path = root / train_val_test
        if path.exists():
            logger.info(f"Deleting existing folder at {path.absolute()}")
        makedirs(path, overwrite=True)
    logger.info(f"Created folders at {root.absolute()}")
    return root


def create_h5_file(savedir, idx, data, compress=True):
    name = "{:0>5}.h5".format(idx)
    path = savedir / name
    compression = "lzf" if compress else None
    with h5py.File(path, "w") as ff:
        ff.create_dataset("frames", data=data["frames"], dtype=np.float32, compression=compression)
        # dtype ubyte is [0, 255]
        ff.create_dataset("labels", data=data["labels"], dtype=np.ubyte, compression=compression)
        ff.create_dataset("bboxs_tlxywh", data=data["bboxs"], dtype=np.ushort, compression=compression)
        ff.create_dataset(
            "frame_times_ms",
            data=data["frame_times_ms"],
            dtype=np.float64,
            compression=compression,
        )
        ff.create_dataset(
            "bbox_times_ms",
            data=data["bbox_times_ms"],
            dtype=np.float64,
            compression=compression,
        )
        ff.create_dataset(
            "bbox_idxs",
            data=data["bbox_idxs"],
            dtype=np.uint32,
            compression=compression,
        )
        ff.create_dataset("speeds", data=data["speeds"], dtype=np.float32, compression=compression)
        ff.create_dataset("stops", data=data["stops"], dtype=np.uint32, compression=compression)


def create_meta_info(ad, root, label_names):
    meta_info = ad.copy()
    meta_info["bbox_format"] = "tlxywh"
    meta_info["labels"] = [int(ll) for ll in label_names]
    path = root / "meta_info.json"
    with open(path, "w") as hd:
        json.dump(meta_info, hd)


def generate_random_move_mnist_dset(ad):
    logger = logging.getLogger("GEN")
    logger.setLevel("INFO")
    root = create_dataset_folders(ad)
    # initialize parameters for sim
    params = RandomMoveMnistGenerator.get_default_params()
    for kk in params.keys():
        params[kk] = ad[kk]
    sim_config = get_config(
        cp=ad["sim.cp"],
        cm=ad["sim.cm"],
        sigma_cp=ad["sim.sigma_cp"],
        sigma_cm=ad["sim.sigma_cm"],
        ref_period_ns=ad["sim.ref_period_ns"],
        log_eps=ad["sim.log_eps"],
    )
    fps_sim = 1000.0 / ad["delta_t_ms"]

    label_names = None
    for tvt_idx, train_val_test in enumerate(["train", "val", "test"]):
        labels_to_use = ad["labels"] if train_val_test == "train" else label_names
        split = 0.9 if train_val_test in ["train", "val"] else 1.0
        if ad["debug"]:
            gen = DebugRandomMoveMnistGenerator(
                ad["time_s"],
                ad["shape"],
                ad["n_samples"][tvt_idx],
                delta_t_ms=ad["delta_t_ms"],
                parameters_sim=params,
                n_objects_per_sample=ad["n_objects_per_sample"],
                labels_to_use=labels_to_use,
                train_val_test=train_val_test,
                split=split,
                seed=3,
            )
        else:
            gen = RandomMoveMnistGenerator(
                ad["time_s"],
                ad["shape"],
                ad["n_samples"][tvt_idx],
                delta_t_ms=ad["delta_t_ms"],
                parameters_sim=params,
                n_objects_per_sample=ad["n_objects_per_sample"],
                labels_to_use=labels_to_use,
                train_val_test=train_val_test,
                split=split,
                seed=3 * (tvt_idx + 1),
            )
        logger.info("Created generator")
        savedir = root / train_val_test
        for ii in ProgressLogger(range(ad["n_samples"][tvt_idx]), name="GEN"):
            seq, bbox_seq, label_seq, time_idxs, speeds, stops = gen.gen_sample()
            # squeeze channel dimension
            seq = seq.squeeze(1)
            if ad["sim.blur_frames"]:
                seq = process_frames(seq)
            events = simulate_events(seq, sim_config, fps_sim, print_progress=False)
            save_events(events, savedir, ii, use_numpy=True)
            logger.info(f"{ii:0>5}: {len(events):0>7} events over {ad['time_s']} s.")
            frame_idxs, frame_times_ms = gen.get_frame_idxs_and_times(ad["frames.fps"])
            data = filter_frames_and_bboxs(
                seq,
                bbox_seq,
                label_seq,
                frame_idxs,
                time_idxs,
                frame_times_ms,
                speeds,
                stops,
                ad["delta_t_ms"],
            )
            create_h5_file(savedir, ii, data, compress=True)
        logger.info(f"Finished creating dset for {train_val_test}")
        if train_val_test == "train":
            label_names = gen.actual_labels
            create_meta_info(ad, root, sorted(label_names))


def filter_frames_and_bboxs(
    seq,
    bbox_seq,
    label_seq,
    frame_idxs,
    time_idxs,
    frame_times_ms,
    speeds,
    stops,
    delta_t_ms,
):
    """Filter data to go from (sim_time, n_objects) to (fps_time, )


    Args:
        seq (np.ndarray): Sequence
        bbox_seq (np.ndarray): (sim_time, n_obj, 4)
        label_seq (np.ndarray): shape (sim_time, n_obj)
        frame_idxs (np.ndarray): shape (sim_time, )
        time_idxs (np.ndarray): shape (sim_time, )
        frame_times_ms (np.ndarray): shape (sim_time, )
        speeds (np.ndarray): shape (sim_time, n_obj, 2)
        stops (np.ndarray): shape (sim_time, n_obj)
        delta_t_ms (float): Simulation time

    Returns:
        List(np.ndarray): Filtered data
    """
    # this is cheating a bit, should rather take moving average of speeds first to get correct results
    bbox_seq = bbox_seq.reshape(-1, 4)
    label_seq = label_seq.ravel()
    speeds = speeds.reshape(-1, 2)
    stops = stops.ravel()
    # times where each single bounding box is shown
    bbox_times_ms = time_idxs * delta_t_ms
    # boxes at t fall in the interval [t - delta_t_ms/2, t + delta_t_ms/2]
    delta_sim_time = delta_t_ms / 2.0
    bbox_start_idxs = np.searchsorted(bbox_times_ms, frame_times_ms - delta_sim_time)
    bbox_stop_idxs = np.searchsorted(bbox_times_ms, frame_times_ms + delta_sim_time)
    data = dict(
        frame_times_ms=frame_times_ms,
        speeds=[],
        stops=[],
        labels=[],
        bboxs=[],
        bbox_times_ms=[],
        bbox_idxs=[],
    )
    for f_idx, (time, start, stop) in enumerate(zip(frame_times_ms, bbox_start_idxs, bbox_stop_idxs)):
        bbox_this = bbox_seq[start:stop]
        data["bboxs"].append(bbox_this)
        data["bbox_times_ms"].append([time] * len(bbox_this))
        data["bbox_idxs"].append([f_idx] * len(bbox_this))
        data["labels"].append(label_seq[start:stop])
        data["speeds"].append(speeds[start:stop])
        data["stops"].append(stops[start:stop])
    for key, val in data.items():
        if key in ["frame_times_ms"]:
            continue
        data[key] = np.concatenate(val)
    data["frames"] = np.stack([seq[idx] for idx in frame_idxs])
    return data


def load_rmmnist_h5(path):
    path = Path(path)
    with h5py.File(path, "r") as hd:
        frames = hd["frames"][:]
        labels = hd["labels"][:]
        bboxs = hd["bboxs_tlxywh"][:]
        frame_times = hd["frame_times_ms"][:]
        bbox_times = hd["bbox_times_ms"][:]
        bbox_idxs = hd["bbox_idxs"][:]
        speeds = hd["speeds"][:]
        stops = hd["stops"][:]
    return frames, labels, bboxs, frame_times, bbox_times, bbox_idxs, speeds, stops


def load_rmmnist_by_index(root, idx, train_val_test="train"):
    root = Path(root)
    meta_info = load_json(root / "meta_info.json")
    path = root / train_val_test / f"{idx:0>5}.h5"
    (
        frames,
        labels,
        bboxs,
        frame_times,
        bbox_times,
        bbox_idxs,
        speeds,
        stops,
    ) = load_rmmnist_h5(path)
    return (
        meta_info,
        frames,
        labels,
        bboxs,
        frame_times,
        bbox_times,
        bbox_idxs,
        speeds,
        stops,
    )


def load_rmmnist_events(root, idx, train_val_test="train", to_float=True):
    root = Path(root)
    path = root / train_val_test / f"{idx:0>5}_events.npy"
    events = np.load(path)
    if to_float:
        events = events_np_to_float(events)
    return events
