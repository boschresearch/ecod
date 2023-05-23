# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
import pprint
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import tqdm
import time

import torch

from pytorch_lightning.callbacks import Callback

from memory_profiler import memory_usage

from ecod.utils.general import ProgressLogger
from ecod.training.mlflow import log_mlflow_artifact


class ProgressLoggerCallback(Callback):
    """Simple callback to print progress after epochs
    Don't inherit from ProgressLoggerBase because don't need updates after every batch
    """

    def __init__(self, len_train=0, len_val=0, len_test=0, log_metrics=True):
        super().__init__()
        self._enabled = True
        self.log_metrics = log_metrics
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test
        self.t_start = time.time()
        self.t_start_epoch = dict(train=None, val=None, test=None)
        self.did_val_this_epoch = False
        self.logger = logging.getLogger("CB-PROGRESS")
        self.logger.setLevel(logging.INFO)
        self.pprinter = pprint.PrettyPrinter(indent=2, width=100, depth=None, stream=None, compact=False)
        self.global_desc = "{name} {epoch}/{max_epoch}"
        self.it_counters = dict(
            train=ProgressLogger(self.len_train, "CB-PROGRESS"),
            val=ProgressLogger(self.len_val, "CB-PROGRESS"),
            test=ProgressLogger(self.len_test, "CB-PROGRESS"),
        )

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def _get_scalar_metrics(self, logs):
        metrics = {}
        other = []
        form = "{:.5g}"
        for key, value in logs.items():
            if hasattr(value, "item") and len(value.shape) == 0:
                metrics[key] = form.format(value.item())
            elif not hasattr(value, "__len__"):
                metrics[key] = form.format(value)
            else:
                other.append(key)
        return metrics, other

    def print(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def _on_epoch_start(self, name, current_epoch, max_epochs):
        if self._enabled:
            self.it_counters[name].start()
            desc = self.global_desc.format(name=name, epoch=current_epoch + 1, max_epoch=max_epochs)
            self.print("Start " + desc)
            self.t_start_epoch[name] = time.time()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.did_val_this_epoch = False
        self._on_epoch_start("train", trainer.current_epoch, trainer.max_epochs)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._on_epoch_start("val", trainer.current_epoch, trainer.max_epochs)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._on_epoch_start("test", trainer.current_epoch, trainer.max_epochs)

    def _on_batch_end(self, name, batch):
        if self._enabled:
            self.it_counters[name].update(1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._on_batch_end("train", batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._on_batch_end("val", batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        self._on_batch_end("test", batch)

    def _on_epoch_end(self, trainer, pl_module, name):
        if self._enabled:
            t_epoch = time.time() - self.t_start_epoch[name]
            self.t_start_epoch[name] = t_epoch
            # want to have train time without val time, but have train_start->val_start->val_end->train_end
            #  so have to subtract val time from train time if validation happened
            if self.did_val_this_epoch and name == "train":
                t_epoch -= self.t_start_epoch["val"]
            pl_module.log(f"{name}_duration", t_epoch, batch_size=1)
            t_tot = time.time() - self.t_start
            pl_module.log(f"total_duration", t_tot, batch_size=1)
            if self.log_metrics:
                metrics, _ = self._get_scalar_metrics(trainer.callback_metrics)
                # get all metrics not related to val or test (e.g., also learning rate) during training
                if name == "train":
                    metrics = {key: val for key, val in metrics.items() if not key.startswith("_")}
                else:
                    # metrics = {key: val for key, val in metrics.items() if key.startswith(name)}
                    metrics = {}
                if len(metrics) > 0:
                    self.print("Metrics:\n" + self.pprinter.pformat(metrics))
            desc = self.global_desc.format(name=name, epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)
            self.print("End " + desc + f", t={t_epoch:.2g}s, t_total={t_tot/3600.:.2g}h")

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        self._on_epoch_end(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.did_val_this_epoch = True
        self._on_epoch_end(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, "test")

    def on_sanity_check_start(self, trainer, pl_module):
        self.disable()

    def on_sanity_check_end(self, trainer, pl_module):
        self.enable()


class BestMetricLoggerCallback(Callback):
    """Callback to save *all* metrics if `metric_name` improved"""

    def __init__(self, metric_name, mode="max"):
        super().__init__()
        self._enabled = True
        self.metric_name = metric_name
        self.log_name = f"best_{self.metric_name}"
        self.mode = mode
        if mode not in ["min", "max"]:
            raise ValueError(f"mode has to be in 'min', 'max', but is {mode}")
        self.best_metrics = {}
        self.best_metric_default = -1e7 if mode == "max" else 1e7

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def on_train_epoch_end(self, trainer, pl_module):
        if self._enabled:
            metric = trainer.callback_metrics.get(self.metric_name, None)
            if metric is None:
                return None
            best_metric = self.best_metrics.get(self.log_name, self.best_metric_default)
            check_max = self.mode == "max" and best_metric < metric
            check_min = self.mode == "min" and best_metric > metric
            if check_max or check_min:
                self.update_and_log_best_metric(self.log_name, metric, pl_module)
                pl_module.log(f"_{self.metric_name}_improved", True, batch_size=1)
                items_list = list(trainer.callback_metrics.items())
                for key, val in items_list:
                    if "best" not in key:
                        log_name = f"_best_{key}"
                        self.update_and_log_best_metric(log_name, val, pl_module)
            else:
                pl_module.log(f"_{self.metric_name}_improved", False, batch_size=1)

    def update_and_log_best_metric(self, metric_name, metric, pl_module):
        if metric is None:
            return None
        self.best_metrics[metric_name] = metric
        pl_module.log(f"{metric_name}", metric, batch_size=1)

    def on_sanity_check_start(self, trainer, pl_module):
        self.disable()

    def on_sanity_check_end(self, trainer, pl_module):
        self.enable()


class SaveBeforeFirstEpochCallback(Callback):
    """Callback to save model before training; most useful to check hparams manually"""

    def __init__(self, best_model_path):
        super().__init__()
        self._enabled = True
        self.best_model_path = best_model_path

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch == 0:
            trainer.save_checkpoint(self.best_model_path)

    def on_sanity_check_start(self, trainer, pl_module):
        self.disable()

    def on_sanity_check_end(self, trainer, pl_module):
        self.enable()


class GPUMemoryCallback(Callback):
    """Callback to log GPU memory"""

    def __init__(self):
        super().__init__()
        self._enabled = True
        self.logger = logging.getLogger("GPUMEM")
        self.logger.setLevel(logging.INFO)

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def _on_epoch_end(self, train_val_test):
        if self._enabled:
            if torch.cuda.is_available():
                gpus = {
                    torch.cuda.get_device_name(ii): torch.cuda.device(ii) for ii in range(torch.cuda.device_count())
                }
                for name, device in gpus.items():
                    memory = torch.cuda.max_memory_allocated(device)
                    self.logger.info(f"Max GPU memory ({train_val_test}) of {name}: {memory / 1024**3:.2f} GB")
                    torch.cuda.reset_peak_memory_stats(device)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        return self._on_epoch_end("train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        return self._on_epoch_end("val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        return self._on_epoch_end("test")

    def on_sanity_check_start(self, trainer, pl_module):
        self.disable()

    def on_sanity_check_end(self, trainer, pl_module):
        self.enable()


class ModelCheckpointMlflow(ModelCheckpoint):
    def _update_best_and_save(self, current: torch.Tensor, trainer, monitor_candidates) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        del_filepath = None
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        log_mlflow_artifact(trainer.logger, filepath)
        logger = logging.getLogger("CHECKPOINT")
        logger.setLevel("INFO")
        logger.info("Logged model checkpoint to mlflow")


class ImprovementTracker:
    def __init__(self, mode):
        self.improved = None
        self.mode = mode
        if mode not in ["min", "max"]:
            raise ValueError(f"mode has to be in 'min', 'max', but is {mode}")
        self.last = -1e7 if mode == "max" else 1e7
        self._enabled = False

    def __call__(self, metric):
        if self._enabled:
            check_max = self.mode == "max" and self.last < metric
            check_min = self.mode == "min" and self.last > metric
            if check_max or check_min:
                self.last = metric
                self.improved = True
            else:
                self.improved = False

    def enable(self):
        self._enabled = True


def get_current_memory():
    return memory_usage(-1, interval=0.0001, timeout=None, max_usage=True)


class MemoryLogger:
    def __init__(self, name=""):
        _name = "MEMORY"
        if len(name) > 0:
            _name += f"_{name.upper()}"
        self.logger = logging.getLogger(_name)
        self.logger.setLevel("INFO")

    @staticmethod
    def get_current_memory():
        return get_current_memory()

    def log(self):
        mem_gb = self.get_current_memory() / 1024
        self.logger.info(f"Current memory: {mem_gb} GB")
