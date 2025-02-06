#!/bin/bash

set -e

if [ -z $1 ]
then
  install_path=~/rocshmem
else
  install_path=$1
fi

src_path=$(dirname "$(realpath $0)")/../../rocshmem/

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$install_path \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DDEBUG=OFF \
    -DPROFILE=OFF \
    -DUSE_GPU_IB=OFF \
    -DUSE_RO=OFF \
    -DUSE_DC=OFF \
    -DUSE_IPC=ON \
    -DUSE_COHERENT_HEAP=ON \
    -DUSE_THREADS=OFF \
    -DUSE_WF_COAL=OFF \
    -DUSE_SINGLE_NODE=ON \
    -DUSE_HOST_SIDE_HDP_FLUSH=OFF \
    -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
    $src_path
cmake --build . --parallel
cmake --install .
