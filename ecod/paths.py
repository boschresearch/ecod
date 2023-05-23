# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

base_dir = Path("")

# MLFLOW
# the folder has to exist already, otherwise you get an OperationalError!
# always provide an absolute path!
mlflow_dir = (base_dir / "").absolute()
mlflow_uri = f"sqlite:///{mlflow_dir}"
tempdir = str(base_dir / "temp")
torch_home = str(base_dir / ".model_weights/torch_checkpoints")

# DATASETS
# MNIST
mnist_path = ""
# RM-MNIST
# move to flash?
random_move_mnist36_root = base_dir / ""
# base_dir / "datasets/rmmnist/mov_mnist_rmov_tms5000_dtmus1000_s0720_s11280_nm1_3-6/"
random_move_mnist36_paths = {
    "train": random_move_mnist36_root / "train",
    "val": random_move_mnist36_root / "val",
    "test": random_move_mnist36_root / "test",
}
random_move_mnist36_meta_info_path = random_move_mnist36_root / "meta_info.json"
# DEBUG
random_move_debug_root = base_dir / ""
# base_dir / "datasets/rmmnist/mov_mnist_rmov_tms5000_dtmus1000_s0720_s11280_nm1_3-6/"
random_move_debug_paths = {
    "train": random_move_debug_root / "train",
    "val": random_move_debug_root / "val",
    "test": random_move_debug_root / "test",
}
random_move_debug_meta_info_path = random_move_debug_root / "meta_info.json"
# PROPHESEE
proph_1mpx_path = Path("")
proph_1mpx_paths = {
    "train": proph_1mpx_path / "train",
    "val": proph_1mpx_path / "val",
    "test": proph_1mpx_path / "test",
}
proph_1mpx_aux_data_path = proph_1mpx_path.parent / "auxiliary_data"


scheduler_logdir = ""
default_queue_name = ""
email = ""
