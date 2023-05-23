#!/bin/bash -e

# This script downloads all necessary libraries for this module
# You can also put the files in `lib` manually

set -e

EIGEN3_VERSION=3.3.9
PYBIND11_VERSION=2.10.3

if ! [ -d ./lib ]; then
    mkdir lib
fi
cd lib/

# Eigen3
eigen_folder=eigen3
if [[ -d "./${eigen_folder}" ]]; then
    rm -r ${eigen_folder}
fi
eigenname="eigen-${EIGEN3_VERSION}.tar.gz"
if [[ ! -f "./${eigenname}" ]]; then
    wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN3_VERSION}/${eigenname}
fi
tar -xf $eigenname
folder=$(find . -maxdepth 1 -type d -name "*eigen*")
mv $folder ${eigen_folder}
echo "Finished setting up Eigen"

# pybind11
pb_folder=pybind11
if [[ -d "./${pb_folder}" ]]; then
    rm -r ${pb_folder}
fi
pybindname="pybind11-${PYBIND11_VERSION}.tar.gz"
if [[ ! -f "./$pybindname" ]]; then
    wget https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz
fi
tar -xf $pybindname
folder=$(find . -maxdepth 1 -type d -name "*pybind11*")
mv $folder ${pb_folder}
echo "Finished setting up pybind"
