cmake_minimum_required(VERSION 3.5)
project(evis)

include_directories(lib/eigen3/)

# build the library
add_subdirectory(lib/pybind11)
pybind11_add_module(trans src/evis/cpp/trans.cpp)
pybind11_add_module(simulator src/evis/cpp/simulator.cpp)
