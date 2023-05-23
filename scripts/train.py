# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Start a training."""
import sys

from ecod.training.prep import start_experiment, parse_args
from ecod.utils.preparation import seed_all, set_torch_home, prepare_logging


def main():
    seed_all(sys.argv)
    set_torch_home()
    prepare_logging()

    args = parse_args()
    start_experiment(args)


if __name__ == "__main__":
    main()
