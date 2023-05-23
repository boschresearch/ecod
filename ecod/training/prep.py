# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import pickle as pkl
from pathlib import Path
import argparse
import logging
import warnings

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_summary import ModelSummary

from ecod.data.lightning import SeqODDataModule
from ecod.training.callbacks import (
    ModelCheckpointMlflow,
    ProgressLoggerCallback,
    BestMetricLoggerCallback,
    GPUMemoryCallback,
)
from ecod.training.mlflow import get_log_root_dir, MLFlowLoggerWrap, log_mlflow_artifact
from ecod.paths import mlflow_uri

from ecod.utils.files import makedirs, get_temp_dir
from ecod.models.lightning import LightningSeqOD


def check_args(args, model, logger):
    if args.load_from:
        nonmatch = {}
        for name in model.hparams:
            if name in ["load_from", "checkpoint_path"]:
                continue
            pold = model.hparams[name]
            pnew = args.__dict__[name]
            if pold != pnew:
                nonmatch[name] = [pold, pnew]
        if nonmatch:
            logger.warn(
                f"Loaded checkpoint but the following args do not match\n"
                + "\n".join([f"{name}: {old} vs {new}" for name, (old, new) in nonmatch.items()]),
            )


def correct_height_width_args(args):
    try:
        shape_t = args.shape_t
    except:
        if args.width > 0:
            if args.height <= 0:
                args.height = args.width
        elif args.height > 0:
            if args.width <= 0:
                args.width = args.height
        else:
            raise ValueError("height or width have to be bigger than 0")


def strip_torch_type(tt):
    return str(tt).replace("<class 'torch.nn.modules.", "").replace("'>", "")


def init_weights(model):
    logger = logging.getLogger("INIT")
    logger.setLevel(logging.INFO)
    param_names = [name for name, _ in model.named_parameters()]
    n_parameters = len(param_names)
    n_init = 0
    temp = "Initialized {} (type {}) with {}".format
    init_str = ""
    for name, mm in model.named_modules():
        if getattr(mm, "is_pretrained", False):
            logger.info(f"Skipping layer {name}, because it is pretrained")
            continue
        if type(mm) in [torch.nn.Conv2d, torch.nn.Linear]:
            init_str += temp(name, strip_torch_type(type(mm)), "kaiming_uniform (weight)")
            torch.nn.init.kaiming_uniform_(mm.weight)
            n_init += 1
            if hasattr(mm, "bias") and mm.bias is not None:
                torch.nn.init.constant_(mm.bias, 0.0)
                init_str += " and zeros (bias)"
                n_init += 1
        elif type(mm) is torch.nn.BatchNorm2d:
            init_str += temp(name, type(mm), "weight=1, bias=0")
            torch.nn.init.constant_(mm.weight, 1.0)
            torch.nn.init.constant_(mm.bias, 0.0)
            n_init += 2
        if len(init_str) > 0 and init_str[-1] != "\n":
            init_str += "\n"
    if n_init == n_parameters:
        logger.info(f"All parameters initialised:\n{init_str}")
    else:
        p_str = str(param_names).replace(" ", "\n")
        logger.info(f"Not all parameters are initialised!!\nInitialised:\n{init_str}All:\n{p_str}")


def get_data_module(args_dict):
    dm = SeqODDataModule(args_dict)
    dm.prepare_data()
    # for now, stage arg is not used in the DataModules of this module
    dm.setup(stage="fit")
    # dm.setup(stage='test')
    return dm


def setup_run(args, logger_name="EVScale"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.info("Running with args:\n{}".format(args))
    return logger


def get_callbacks(args):
    callbacks = {}
    if not args.fast_dev_run:
        monitor = "val_mAP"
        best_model_name = "best_model"
        checkpoint_callback = ModelCheckpointMlflow(
            dirpath=args.temp_dir,
            monitor=monitor,
            mode="max",
            filename=best_model_name,
        )
        callbacks["checkpoint"] = checkpoint_callback
        early_stopping_callback = EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=0.001,
            verbose=True,
        )
        callbacks["early_stopping"] = early_stopping_callback
        callbacks["best_metric"] = BestMetricLoggerCallback("val_mAP", "max")
        callbacks["model_summary"] = ModelSummary(max_depth=20)
        callbacks["gpu_mem"] = GPUMemoryCallback()
        # bmp = callbacks['checkpoint'].format_checkpoint_name({})
        # callbacks['save_before_first_epoch'] = SaveBeforeFirstEpochCallback(bmp)
    return callbacks


def add_checkpoint_callbacks(args, dm, callbacks):
    if not args.fast_dev_run:
        len_train = len(dm.train_dataloader())
        len_val = len(dm.val_dataloader())
        len_test = len(dm.test_dataloader())
        progress_callback = ProgressLoggerCallback(len_train, len_val, len_test)
        callbacks["progress"] = progress_callback
    return callbacks


def get_profiler(args_dict):
    if args_dict.get("profiler", None) is None:
        return None
    prof_name = args_dict["profiler"]
    temp_dir = args_dict["temp_dir"]
    if prof_name == "advanced":
        profiler = pl.profiler.advanced.AdvancedProfiler(dirpath=temp_dir, filename="profile")
    elif prof_name == "simple":
        profiler = pl.profiler.simple.SimpleProfiler(dirpath=temp_dir, filename="profile")
    elif prof_name == "pytorch":
        profiler = pl.profiler.pytorch.PytorchProfiler(dirpath=temp_dir, filename="profile")
    else:
        raise ValueError(f"--profiler has to be in ['simple', 'advanced', 'pytorch', None] but is {prof_name}")
    return profiler


def setup_experiment(args):
    logger = setup_run(args, "EVScale")
    log_dir = get_log_root_dir()
    makedirs(log_dir, overwrite=False)
    if args.temp_dir is None:
        args.temp_dir = str(get_temp_dir(log_dir))
    with open(Path(args.temp_dir) / "args_dict.pkl", "wb") as hd:
        pkl.dump(vars(args), hd)
    # this is only in here because mlflow with sqlite always creates a mlruns folder in the working directory and
    #  if it is in the temporary directory, it is less annoying than in the actual working directory
    os.chdir(args.temp_dir)
    logger.info(f"Log dir: {log_dir}, temp dir: {args.temp_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch_size is -1 but autoscaling does not work right now. Choose a batch_size > 0")
        auto_scale_batch_size = True
        args.batch_size = 2
    else:
        auto_scale_batch_size = False
    callbacks = get_callbacks(args)
    if args.load_from is not None:
        model = LightningSeqOD.load_from_checkpoint(args.load_from)
    else:
        model = LightningSeqOD(vars(args))
        init_weights(model)
    if args.weights_path is not None:
        state_dict = torch.load(args.weights_path)["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded weights from {args.weights_path}")
    check_args(args, model, logger)
    args_dict = model.hparams.copy()
    with open(Path(args.temp_dir) / "args_dict.pkl", "wb") as hd:
        pkl.dump(args_dict, hd)
    dm = get_data_module(args_dict)
    # can add some callbacks only after having dm, but also need to init model before dm to get model-specific args
    callbacks = add_checkpoint_callbacks(args, dm, callbacks)
    mlflow_logger = MLFlowLoggerWrap(
        f"{args.dataset}-{args.name}",
        tracking_uri=f"{mlflow_uri}",
    )
    profiler = get_profiler(vars(args))
    gpus = (
        torch.cuda.device_count()
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_name(0) == "Tesla V100-SXM2-32GB"
            or torch.cuda.get_device_name(0).startswith("NVIDIA A100")
        )
        else 0
    )
    if gpus == 0:
        raise RuntimeError("No GPUs found.")
    if gpus > 1:
        raise RuntimeError(
            "For now, can only train with 1 GPU. See "
            "https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html to adapt "
            "the code"
        )
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=gpus,
        enable_progress_bar=False,
        # only set benchmark when batch size doesnt change,
        #  but with od, batch size is dependent on proposed regions
        benchmark=False,
        default_root_dir=args.temp_dir,
        auto_scale_batch_size=auto_scale_batch_size,
        log_every_n_steps=20,
        # auto-finding lr heuristically leads to worse results...
        auto_lr_find=False,
        gradient_clip_val=1.0,
        callbacks=list(callbacks.values()),
        logger=mlflow_logger,
        profiler=profiler,
    )
    return trainer, model, dm, callbacks


def start_experiment(args):
    trainer, model, dm, callbacks = setup_experiment(args)
    if not args.test:
        trainer.tune(model, datamodule=dm)
        trainer.fit(model, datamodule=dm)
        if not args.fast_dev_run:
            best_model_path = callbacks["checkpoint"].best_model_path
            if Path(best_model_path).exists() and len(best_model_path) > 0:
                log_mlflow_artifact(trainer.logger, best_model_path)
                model = LightningSeqOD.load_from_checkpoint(best_model_path)
            else:
                raise RuntimeError(
                    "The path to the best model should exist at this point, but does not exist: "
                    f"{callbacks['checkpoint'].best_model_path}. Did at least one epoch finish?"
                )
    else:
        trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


def get_start_parser(args):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--weights_seed", default=-1)
    parser.add_argument("--random", action="store_true")
    parser.add_argument(
        "--use_install",
        action="store_true",
        help="Use installed module instead of current",
    )

    parser = LightningSeqOD.add_model_specific_args(parser)
    parser = SeqODDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # parser.set_defaults(progress_bar_refresh_rate=0)
    return parser


def fix_start_parser_args(args):
    correct_height_width_args(args)
    if args.shape_t is None:
        args.shape_t = [args.n_timesteps, args.n_channels, args.height, args.width]
    else:
        args.n_timesteps, args.n_channels, args.height, args.width = args.shape_t
    if args.train_n_predictions is None:
        args.train_n_predictions = args.shape_t[0]
    elif args.train_n_predictions > args.shape_t[0]:
        raise ValueError(
            f"train_n_predictions has to be <= args.n_timesteps, "
            f"but are {args.train_n_predictions}, {args.n_timesteps}"
        )
    if args.test:
        args.max_epochs = 1
    if hasattr(args, "prior_not_clip"):
        args.prior_clip = not args.prior_not_clip
    if args.prior_scales is not None:
        args.prior_min_sizes = args.prior_scales
    if hasattr(args, "boxes_to_locations"):
        args.boxes_to_locations = args.bbox_loss in ["smooth_l1"]
    if len(args.prior_aspect_ratios) == 1:
        args.prior_aspect_ratios = args.prior_aspect_ratios * len(args.hidden_dims)
    return args


def parse_args(args=None):
    parser = get_start_parser(args)
    args = parser.parse_args(args=args)
    args = fix_start_parser_args(args)
    return args
