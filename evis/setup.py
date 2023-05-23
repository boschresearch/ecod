# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from os import environ
from pathlib import Path
import sys
import subprocess
from shutil import rmtree
from warnings import warn

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path.cwd())


class CMakeBuild(build_ext):
    def run(self):
        if Path(self.build_temp).exists():
            warn(f"Build directory at {self.build_temp} should not yet exist, please delete manually")
            rmtree(self.build_temp)
        for ext in self.extensions:
            self.build_extension(ext)
        if Path(self.build_temp).exists():
            rmtree(self.build_temp)

    def build_extension(self, ext):
        build_type = "Debug" if self.debug else "Release"
        cmake_call = [
            "cmake",
            ext.sourcedir,
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(Path(self.get_ext_fullpath(ext.name)).parent.absolute())}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        cmake_build_call = ["cmake", "--build", ".", "--config", build_type, "--", "-j2"]
        old_cxx_flags = environ.get("CXXFLAGS", "")
        environ["CXXFLAGS"] = f'{old_cxx_flags} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        Path(self.build_temp).mkdir(exist_ok=True, parents=True)
        try:
            subprocess.check_call(cmake_call, cwd=self.build_temp)
            subprocess.check_call(cmake_build_call, cwd=self.build_temp)
        finally:
            environ["CXXFLAGS"] = old_cxx_flags


setup(
    name="evis",
    version="0.1",
    author="Alexander Kugele",
    author_email="alexander.kugele@de.bosch.com",
    description="Process event camera data efficiently.",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("evis.trans"), CMakeExtension("evis.simulator")],
    cmdclass=dict(build_ext=CMakeBuild),
    test_suite="tests",
    zip_safe=False,
)
