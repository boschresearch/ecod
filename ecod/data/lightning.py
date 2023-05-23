# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse

import torch

import pytorch_lightning as pl

from ecod.data.rmmnist.dataset import RandomMoveMnistOD
from ecod.data.proph1mpx.dataset import Proph1MpxOD


class CatBoxesCollator:
    def __call__(self, batch):
        outs = []
        for ii, ts in enumerate(zip(*batch)):
            if ii == 0:
                outs.append(torch.stack(ts))
            else:
                outs.append(torch.cat(ts))
        return outs


class IgnoreIndexCollator:
    def __call__(self, batch):
        outs = []
        for ii, ts in enumerate(zip(*batch)):
            if ii < len(batch[0]) - 1:
                outs.append(torch.stack(ts))
        return outs


class SeqODDataModule(pl.LightningDataModule):
    def __init__(self, args_dict):
        super().__init__()
        self.args_dict = args_dict
        # (time, channels, h, w)
        self.shape = args_dict["shape_t"][-2:]
        self.n_seqs = args_dict["shape_t"][0]
        # for auto-tuning needs to be member
        self.batch_size = args_dict["batch_size"]
        self.pin_memory = not args_dict["not_pin_memory"]
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None

    def prepare_data(self):
        pass

    def get_batch_size(self, dset):
        return self.batch_size if len(dset) > self.batch_size else len(dset)

    def select_dset(self):
        if self.args_dict["dataset"] in ["random_move_mnist36_od", "random_move_debug_od"]:
            train_dset = RandomMoveMnistOD(self.args_dict, "train")
            val_dset = RandomMoveMnistOD(self.args_dict, "val")
            test_dset = RandomMoveMnistOD(self.args_dict, "test")
        elif self.args_dict["dataset"] == "proph_1mpx":
            train_dset = Proph1MpxOD(self.args_dict, "train", bbox_suffix=self.args_dict["bbox_suffix_train"])
            val_dset = Proph1MpxOD(self.args_dict, "val", bbox_suffix=self.args_dict["bbox_suffix_test"])
            test_dset = Proph1MpxOD(self.args_dict, "test", bbox_suffix=self.args_dict["bbox_suffix_test"])
        else:
            raise ValueError(
                "dataset name has to be in 'proph_1mpx', 'random_move_mnist36_od', "
                f"but is {self.args_dict['dataset']}"
            )
        return train_dset, val_dset, test_dset

    def setup(self, fit=None, stage=None):
        if self.train_dset is None or self.val_dset is None or self.test_dset is None:
            self.train_dset, self.val_dset, self.test_dset = self.select_dset()

    def train_dataloader(self):
        batch_size = self.get_batch_size(self.train_dset)
        shuffle = False if self.args_dict["test_dset_is_train_dset"] else True
        return torch.utils.data.DataLoader(
            self.train_dset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.args_dict["num_workers"],
            # collate_fn=IgnoreIndexCollator(),
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # batch_size = self.get_batch_size(self.val_dset)
        batch_size = 1
        return torch.utils.data.DataLoader(
            self.val_dset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args_dict["num_workers"],
            collate_fn=CatBoxesCollator(),
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        # batch_size = self.get_batch_size(self.test_dset)
        batch_size = 1
        return torch.utils.data.DataLoader(
            self.test_dset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args_dict["num_workers"],
            collate_fn=CatBoxesCollator(),
            pin_memory=self.pin_memory,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", default=-1, type=int)
        parser.add_argument(
            "--dataset",
            default="random_move_debug_od",
            choices=[
                "random_move_mnist36_od",
                "random_move_debug_od",
                "proph_1mpx",
            ],
        )
        parser.add_argument("--num_workers", default=6, type=int)
        parser.add_argument("--not_pin_memory", action="store_true")
        # debug training by using same dset for train, val and test. Can't use --overfit_batches because the
        #  transformations are different for train vs val and test.
        parser.add_argument("--test_dset_is_train_dset", action="store_true")
        # voxel grid args
        parser.add_argument("--n_bins", default=5, type=int)
        # random_move_mnist args
        parser.add_argument("--random_move_mnist_frames", action="store_true")
        parser.add_argument("--random_move_mnist_n_objects", type=int, default=1)
        # bbox file args
        parser.add_argument(
            "--bbox_suffix_train",
            type=str,
            default="none",
            choices=[
                "none",
                "only_moving",
                "only_moving_diff",
                "filtered",
                "only_moving0",
                "only_moving10",
                "only_moving100",
                "only_moving1000",
                "only_moving10000",
            ],
            help="Suffix to add to bounding box path. Can load different filtered versions.",
        )
        parser.add_argument(
            "--bbox_suffix_test",
            type=str,
            default="none",
            choices=[
                "none",
                "only_moving",
                "only_moving_diff",
                "filtered",
                "only_moving0",
                "only_moving10",
                "only_moving100",
                "only_moving1000",
                "only_moving10000",
            ],
            help="Suffix to add to bounding box path. Can load different filtered versions.",
        )
        return parser
