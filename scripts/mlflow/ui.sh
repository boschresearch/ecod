#!/bin/bash

if [[ -z $MLFLOW_TRACKING_URI ]]; then
    echo "Can't start mlflow ui because MLFLOW_TRACKING_URI or MLFLOW_DEFAULT_ARTIFACT_ROOT is not set"
    exit 1
fi
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT --port 43458
