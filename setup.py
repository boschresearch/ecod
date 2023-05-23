# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from setuptools import setup, find_packages

setup(
    name="ecod",
    version="0.1",
    description="Event camera object detection with low event count filtering",
    author="Alexander Kugele",
    author_email="alexander.kugele@de.bosch.com",
    packages=find_packages(),
)
