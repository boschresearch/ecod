# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import shutil
import json
import numpy as np
import pickle as pkl
import h5py

import torch

def makedirs(path, overwrite=False, remove_empty=False):
    path = Path(path)
    if path.exists():
        if overwrite:
            shutil.rmtree(str(path))
        else:
            if remove_empty:
                try:
                    path.rmdir()
                except OSError:
                    raise OSError("Save directory is not empty and overwrite is set to False")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_json(path):
    path = Path(path)
    with open(path, "r") as hd:
        meta_info = json.load(hd)
    return meta_info


def load_h5_by_keys(path, keys):
    path = Path(path)
    data = {}
    with h5py.File(path, "r") as hd:
        for key in keys:
            data[key] = hd[key][:]
    return data


def convert_pkl_to_h5(data_path):
    path = Path(data_path)
    path_h5 = (path.parent.parent / "h5" / path.name).with_suffix(".h5")
    path_h5 = path_h5.parent / path_h5.name.replace("atis_", "ncars_")
    makedirs(path_h5.parent, overwrite=False)
    with open(data_path, "rb") as hd:
        data = pkl.load(hd)
    labels = np.array(data["labels"])
    with h5py.File(path_h5, "w") as ff:
        ff.create_dataset("labels", data=labels, dtype=np.uint32)
        for ii, event_array in enumerate(data["data"]):
            # pkl data has mus, x, y, [-1,1]
            ff.create_dataset(f"{ii:0>5}", data=event_array.astype(np.float32), dtype=np.float32)


def get_temp_dir(log_root_dir):
    """Temp dir during experiment. Don't forget to log artifacts with mlflow

    Args:
        log_root_dir:

    Returns:

    """
    import tempfile

    temp_dir = Path(log_root_dir) / "temp"
    makedirs(temp_dir, overwrite=False)
    temp_dir = Path(tempfile.mkdtemp(dir=temp_dir))
    return temp_dir


def value_list_type(inp_str):
    try:
        output = json.loads(inp_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not decode string {inp_str} to a list. If strings are inside the list,"
            f" they have to be escaped. Original error: {e}"
        )
    return output


def load_args_dict(exp_dir):
    paths = [pp for pp in Path(exp_dir).glob("best_model*ckpt")]
    if len(paths) != 1:
        raise RuntimeError(
            f"There should be exactly one checkpoint with the best model in this directory, " f"but found {paths}"
        )
    path = paths[0]
    args_dict = torch.load(path, map_location="cpu")["hyper_parameters"]
    return args_dict
