#!/bin/bash

echo ""
echo "You have to source this script from the ecod folder: -> source ./scripts/mlflow/set_env.sh <-"

MLFLOW_TRACKING_URI=$(python -c "import sys; sys.path.insert(0, '.'); import ecod.paths as p; print(p.mlflow_uri)")
MLFLOW_DEFAULT_ARTIFACT_ROOT=$(python -c "import sys; sys.path.insert(0, '.'); import ecod.training.mlflow as p; print(p.get_mlflow_artifact_dir(p.get_log_root_dir()))")

echo "Setting tracking: ${MLFLOW_TRACKING_URI}"
echo "Setting artifact root: ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"

export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
export MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT=600
# custom var for this repo, nothing mlflow provides out of the box
export MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT
