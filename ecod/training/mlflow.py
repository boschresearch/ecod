"""MLFlow-related functions.

PyTorch MLFlowlogger slightly adapted from
https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/loggers/mlflow.py
Apache License 2.0  Copyright 2018-2021 William Falcon
"""
# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import pickle as pkl
from pathlib import Path

import mlflow
from mlflow.entities import ViewType

import torch

from pytorch_lightning import loggers as pl_loggers, _logger as log
from pytorch_lightning.loggers.logger import rank_zero_experiment

from ecod.paths import mlflow_uri, mlflow_dir


def get_mlflow_log_dir(trainer, fast_dev_run):
    if fast_dev_run:
        log = None
    else:
        experiment = trainer.logger.experiment.get_experiment(trainer.logger.experiment_id)
        log = Path(experiment.artifact_location)
    return log


def log_mlflow_artifact(logger, local_path):
    logger.experiment.log_artifact(run_id=logger.run_id, local_path=local_path)


def set_mlflow_tracking():
    mlflow.set_tracking_uri(f"{mlflow_uri}")


def filter_experiments(
    experiment_ids=None,
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=100000,
    order_by=None,
    output_format="pandas",
):
    """Filter mlflow experiments

    Args:
        experiment_ids (List[Int], optional): List of experiment ids to check. Defaults to None.
        filter_string (str, optional): Check https://mlflow.org/docs/latest/python_api/mlflow.html . Defaults to ''.
        run_view_type ([type], optional):  Defaults to active runs.
        max_results (int, optional): Defaults to 100000.
        order_by ([type], optional): Defaults to None.
        output_format (str, optional): Defaults to 'pandas'.

    Returns:
        pandas.DataFrame: DataFrame with all results
    """
    if experiment_ids is None:
        experiment_ids = [exp.experiment_id for exp in mlflow.tracking.client.MlflowClient().list_experiments()]
    df = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=run_view_type,
        max_results=max_results,
        order_by=order_by,
        output_format=output_format,
    )
    return df


def get_log_root_dir():

    if mlflow_uri.startswith("sqlite"):
        path = Path("/" + mlflow_uri.lstrip("sqlite:/"))
        return str(path.parent)
    elif mlflow_uri.startswith("file"):
        path = Path("/" + mlflow_uri.lstrip("file:/"))
        return str(path)
    else:
        raise ValueError(f"Can't get root dir for {mlflow_uri}")


def get_mlflow_artifact_dir(save_dir):
    return str(Path(save_dir) / "artifacts")


class MLFlowLoggerWrap(pl_loggers.MLFlowLogger):
    @property
    def save_dir(self):
        return get_log_root_dir()

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment_id is None:
            experiment = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if experiment is not None:
                self._experiment_id = experiment.experiment_id
            else:
                log.warning(f"Experiment with name {self._experiment_name} not found. Creating it.")
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name,
                    artifact_location=get_mlflow_artifact_dir(self.save_dir),
                )
                # TODO: Tried to set artifact location automatically, however, does not use the same folders that
                #  pytorch-lightning uses
                # import os
                # cwd = os.getcwd()
                # os.chdir(self.save_dir)
                # self._experiment_id = self._mlflow_client.create_experiment(name=self._experiment_name,)
                # os.chdir(cwd)

        if self._run_id is None:
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=self.tags)
            self._run_id = run.info.run_id
        return self._mlflow_client


def get_run_config_and_artifact_uri(run_id):
    exp_df = filter_experiments(experiment_ids=None, filter_string="")
    run_config = exp_df[exp_df["run_id"].str.startswith(run_id)]
    artifact_uri = Path(run_config["artifact_uri"].values[0])
    return run_config, artifact_uri


def get_args_dict(artifact_uri):
    state_dict = torch.load(artifact_uri / "best_model.ckpt", map_location=torch.device("cpu"))
    args_dict = state_dict["hyper_parameters"]
    return args_dict


def mlflow_name(name, dataset):
    return f"{dataset}-{name}"


def mlflow_create_or_get_experiment(name, dataset):
    mlflow_logger = MLFlowLoggerWrap(mlflow_name(name, dataset), tracking_uri=f"{mlflow_uri}")
    return mlflow_logger.experiment


def mlflow_create_or_get_experiment_id(name, dataset):
    name = mlflow_name(name, dataset)
    savedir = mlflow_dir.parent
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=get_mlflow_artifact_dir(savedir))
    else:
        mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    return experiment.experiment_id
