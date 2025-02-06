#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
PYNVSHMEM_DIR=${SCRIPT_DIR}/../
NVSHMEM_ROOT=${SCRIPT_DIR}/../../nvshmem/build/install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_ROOT}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export PYTHONPATH=$PYTHONPATH:${PYNVSHMEM_DIR}/build
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
torchrun --nproc_per_node=8 --nnodes=1 pynvshmem/example/run_ring_put.py
