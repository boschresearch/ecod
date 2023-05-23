# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Generate a dataset of moving MNIST numbers. Frames and events.

To generate the RM-MNIST dataset used in the paper:
`python scripts/data/rmmnist_generate.py --savedir <savedir>`

To generate a small debug dataset:
`python scripts/data/rmmnist_generate.py --savedir <savedir> --debug`
"""
import sys
import argparse

from ecod.utils.preparation import seed_all, set_torch_home, prepare_logging
from ecod.data.sim.rmmnist import generate_random_move_mnist_dset


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--weights_seed", default=-1)
    parser.add_argument("--random", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="If true, generate a small pre-defined dataset (ignore most cmd args)"
    )
    # save args
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, nargs=3, default=[50, 10, 10])

    # gen args
    parser.add_argument("--time_s", type=float, default=5.0)
    parser.add_argument("--delta_t_ms", type=float, default=1.0)
    parser.add_argument("--shape", type=int, nargs=2, default=(720, 1280))
    parser.add_argument("--n_objects_per_sample", type=int, default=1)
    parser.add_argument("--frames.fps", type=float, default=60.0)
    # random movements params
    parser.add_argument("--move_rate_hz", type=float, default=0.5)
    parser.add_argument("--move_time_ms", type=float, default=2000.0)
    parser.add_argument("--labels", type=int, nargs="+", default=[3, 6])
    # sim params
    parser.add_argument("--sim.cp", type=float, default=1.0)
    parser.add_argument("--sim.cm", type=float, default=1.0)
    parser.add_argument("--sim.sigma_cp", type=float, default=0.0)
    parser.add_argument("--sim.sigma_cm", type=float, default=0.0)
    parser.add_argument("--sim.ref_period_ns", type=int, default=int(1e5))
    parser.add_argument("--sim.log_eps", type=float, default=0.001)
    parser.add_argument("--sim.blur_frames", action="store_true")

    args = parser.parse_args(args)
    return vars(args)


def main():
    seed_all(sys.argv)
    set_torch_home()
    prepare_logging()

    ad = parse_args()
    if ad["debug"]:
        ad["time_s"] = 1.0
        ad["n_samples"] = [6, 6, 6]
        ad["delta_t_ms"] = 1.0
        ad["n_objects_per_sample"] = 1
        ad["labels"] = [3, 6]
        ad["seed"] = 1
        ad["move_rate_hz"] = 0.5
        ad["move_time_ms"] = ad["time_s"] * 1000.0
    generate_random_move_mnist_dset(ad)


if __name__ == "__main__":
    main()
