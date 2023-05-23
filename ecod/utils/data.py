# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

import torch
from torch.utils.data import DataLoader

from ecod.utils.files import load_json
from ecod.paths import random_move_mnist36_meta_info_path, random_move_debug_meta_info_path


def get_idxs(n_samples, shuffle=True, seed=215, fraction=1.0, is_val=False):
    idxs = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idxs)
    n_frac = int(fraction * n_samples)
    if is_val:
        idxs = idxs[n_frac:]
    else:
        idxs = idxs[:n_frac]
    return idxs


def get_dataset_attributes(name: str):
    if name.lower() in ["proph_1mpx", "proph_1mpx"]:
        attrs = dict(
            shape_max=(720, 1280),
            time_mean=60000.0,
            time_std=0.0,
            # time between samples; be careful: there are not always gt boxes every delta_t microseconds
            delta_t_mus=16682.47834,
            delta_t_var_mus=4,
            class_names=(
                "__background__",
                "pedestrian",
                "two wheeler",
                "car",
            ),
            n_classes=4,
            class_names_full=(
                "__background__",
                "pedestrian",
                "two wheeler",
                "car",
                "truck",
                "bus",
                "traffic sign",
                "traffic light",
            ),
            n_classes_full=8,
            channels=2,
        )
    elif name.lower() in ["mnist"]:
        attrs = dict(
            shape_max=(28, 28),
            class_names=[str(ii) for ii in range(10)],
            mean_int=33.318421,
            std_int=78.56749,
            n_classes=10,
            channels=1,
        )
    elif name.lower() in [
        "shapes",
        "shapes_translation",
        "shapes30",
        "shapes90",
        "shapes_simple",
    ]:
        attrs = dict(
            shape_max=(180, 240),
            time_mean=44.065,
            time_std=0.0,
            class_names=(
                "__background__",
                "star",
                "triangle",
                "arrow",
                "lshape",
                "bar",
                "ellipse",
                "wave",
                "pacman",
                "hexagon",
                "circle",
            ),
            n_classes=11,
            channels=2,
        )
    elif name.lower() in ["random_move_mnist36_od", "random_move_debug_od"]:
        if name.lower() == "random_move_mnist36_od":
            meta_info = load_json(random_move_mnist36_meta_info_path)
        else:
            meta_info = load_json(random_move_debug_meta_info_path)
        class_names = ["__background__", *[str(ii) for ii in meta_info["labels"]]]
        attrs = dict(
            shape_max=meta_info["shape"],
            class_names=class_names,
            n_classes=len(class_names),
            channels=2,
            delta_t_mus=16666.666666666668,
        )
    else:
        raise ValueError(f"Cannot get attributes for name '{name}'")
    return attrs


def pad_channel_with_zeros(frames, n_channels):
    n_dims = len(frames.shape)
    if n_dims == 4:
        pad_channel = 1
        pad = np.zeros((frames.shape[0], n_channels, *frames.shape[2:]))
    else:
        pad_channel = 0
        pad = np.zeros((n_channels, *frames.shape[1:]))
    return np.concatenate([frames, pad], pad_channel)


def make_data_loader(
    dataset,
    num_workers,
    batch_size,
    pin_memory=True,
    is_train=True,
    shuffle=True,
    drop_last=False,
    **kwargs,
):
    """ """
    shuffle = shuffle and is_train
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )
    return data_loader


def identity_transform(*args, **kwargs):
    return args


def add_fake_bins_and_chans(seq, n_bins, n_chans):
    """
    For tensor of shape [a, b, c] repeat entries to create tensor of shape [a, n_bins, n_chans, b, c]
    """
    return torch.movedim(
        seq.repeat(n_bins, n_chans, 1).reshape(n_bins, seq.shape[0], n_chans, *seq.shape[1:]),
        0,
        1,
    )
