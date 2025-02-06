#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
PYROCSHMEM_DIR=${SCRIPT_DIR}/../
ROCSHMEM_ROOT=${SCRIPT_DIR}/../../rocshmem_build/install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROCSHMEM_ROOT}/lib

export PYTHONPATH=$PYTHONPATH:${PYROCSHMEM_DIR}/build
python3 naive.py
# torchrun --nproc_per_node=8 --nnodes=1 naive.py
