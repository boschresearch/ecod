# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np

from ecod.paths import mnist_path
from ecod.utils.data import get_idxs


def load_mnist_data(split=0.9, train_val_test="train", shuffle=True):
    data = np.load(mnist_path)
    if train_val_test in ["train", "val"]:
        imgs = data["x_train"]
        labels = data["y_train"]
        idxs = get_idxs(
            len(labels),
            fraction=split,
            is_val=train_val_test == "val",
            shuffle=shuffle,
            seed=12,
        )
        imgs = imgs[idxs]
        labels = labels[idxs]
    else:
        imgs = data["x_test"]
        labels = data["y_test"]
    return imgs, labels
