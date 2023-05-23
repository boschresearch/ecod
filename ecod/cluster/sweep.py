# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import tempfile
import logging
import subprocess
from pathlib import Path
from string import Template

from ecod.paths import scheduler_logdir, default_queue_name, email, tempdir
from ecod.utils.files import makedirs, get_temp_dir
from ecod.training.mlflow import mlflow_create_or_get_experiment, get_log_root_dir


def add_default_script_args(
    params,
    name,
    memory_gb=8,
    num_workers=4,
    runtime_hours=24,
    queue=default_queue_name,
    n_gpus=1,
    mig_slices=2,
    email=email,
    scheduler_logdir=scheduler_logdir,
    conda_version="4.11.0",
):
    _params = {
        "--xsr_name": name,
        "--xsr_logdir": scheduler_logdir,
        "--xsr_runtime_hours": runtime_hours,
        "--xsr_memory_mb": memory_gb * 1024,
        "--xsr_num_workers": num_workers,
        "--xsr_queue": queue,
        "--xsr_n_gpus": n_gpus,
        "--xsr_mig_slices": mig_slices,
        "--xsr_email": email,
        "--xsr_conda_version": conda_version,
    }
    params.update(_params)
    return params


def get_scheduler_kwargs(params):
    scheduler_kwargs = {}
    for key, value in params.items():
        if key.startswith("--xsr_") and key not in ["xsr_n_gpus", "xsr_mig_slices"]:
            scheduler_kwargs[key[6:]] = str(value)
    gpu_string = f"num={params['--xsr_n_gpus']}"
    if params["--xsr_queue"] in ["inter_a100", "batch_a100_mig"]:
        gpu_string += f":mig={params['--xsr_mig_slices']}"
    scheduler_kwargs["gpu_string"] = gpu_string
    return scheduler_kwargs


def params_to_script_args(params):
    script_args = ""
    for key, value in params.items():
        if (not key.startswith("--xsr_") and key not in ["hpc_exp_number"]) or key in [
            "--xsr_name",
            "--xsr_num_workers",
        ]:
            # patch num_workers only if it is not overridden by --num_workers
            if key == "--xsr_num_workers":
                if "--num_workers" not in params.keys():
                    key = f"--{key[6:]}"
                else:
                    continue
            elif key.startswith("--xsr_"):
                key = f"--{key[6:]}"
            if value is True:
                script_args += f"{key} "
            elif value in [False, None]:
                continue
            else:
                script_args += f"{key} {str(value)} "
    # remove trailing whitespace
    return script_args[:-1]


def get_param_str(key, val):
    if val is True:
        return f"{key}"
    elif val is False:
        return ""
    else:
        return f"{key} {val}"


def append_params_str(params_str, key, val):
    param = get_param_str(key, val)
    if len(param) > 0:
        params_str = params_str + " " + param
    return params_str


def get_params_list(params):
    params_list = [""]
    for key, value in params.items():
        if not isinstance(value, str) and hasattr(value, "__len__"):
            params_list_add = []
            for pp in params_list:
                for val in value:
                    params_str = append_params_str(pp, key, val)
                    params_list_add.append(params_str)
            params_list = params_list_add
        else:
            for ii, pp in enumerate(params_list):
                params_list[ii] = append_params_str(pp, key, value)
    return params_list


def get_params_dict_list(params):
    params_dict_list = [{}]
    for key, value in params.items():
        if not isinstance(value, str) and hasattr(value, "__len__"):
            params_list_add = []
            for pp in params_dict_list:
                for val in value:
                    pc = pp.copy()
                    pc[key] = val
                    params_list_add.append(pc)
            params_dict_list = params_list_add
        else:
            for ii, pp in enumerate(params_dict_list):
                params_dict_list[ii][key] = value
    return params_dict_list


def submit_jobs(hparams_list, root_dir, start_job=True, print_start=True, env=None):
    with open(root_dir / "cluster/run.bsub", "r") as hd:
        bsub_template = Template("".join(hd.readlines()))
    logger = logging.getLogger("SUBMIT")
    logger.setLevel("INFO")
    log_dir = get_log_root_dir()
    makedirs(log_dir, overwrite=False)
    for hparams in hparams_list:
        temp_dir = get_temp_dir(log_dir)
        hparams["--temp_dir"] = str(temp_dir)
        sched_kwargs = get_scheduler_kwargs(hparams)
        script_args = params_to_script_args(hparams)
        script = bsub_template.safe_substitute(root_dir=root_dir, script_args=script_args, **sched_kwargs)
        path = temp_dir / "run.bsub"
        with open(str(path), "w") as f:
            f.write(script)
        logger.info(f"Saving submit script at {temp_dir}")
        if start_job:
            with open(str(path), "r") as hd:
                pipe = subprocess.Popen(
                    ["bsub"],
                    stdin=hd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
        if print_start:
            logger.info("Started job with parameters:")
            logger.info(f"dict: {hparams}")
            logger.info(f"str: {script_args}")


def setup_hparams_parser(params):
    # important to create experiment here, because else could happen that multiple runs try to create the same
    # experiment
    name = params["--xsr_name"]
    dataset = params["--dataset"]
    mlflow_create_or_get_experiment(name, dataset)
    hparams_list = get_params_dict_list(params)
    return hparams_list


def start_sweep(params, n_repeats, start_jobs=True, env=None):
    root_dir = Path(__file__).absolute().parents[2]
    hparams_list = setup_hparams_parser(params) * n_repeats
    submit_jobs(hparams_list, root_dir, start_jobs, env=env)
