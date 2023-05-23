# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import sys
import os
import numpy as np
import logging

from lightning_fabric.utilities.seed import seed_everything

from ecod.paths import torch_home


def seed_all(args):
    if "--random" in args:
        seed = np.random.randint(51241245)
        sys.stderr.write("seed IS RANDOM (seed={})\n".format(seed))
    else:
        if "--weights_seed" in args:
            seed = int(args[args.index("--weights_seed") + 1])
        else:
            seed = 1
        sys.stderr.write("seed IS FIXED (seed={})\n".format(seed))
    seed_everything(seed)


def set_torch_home(home=torch_home):
    os.environ["TORCH_HOME"] = home


def prepare_logging():
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s:%(levelname)s:%(name)s: %(message)s",
        datefmt="%y%m%d-%H:%M:%S",
    )
