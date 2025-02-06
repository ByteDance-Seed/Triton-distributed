#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
TRITON_ROCSHMEM_DIR=${SCRIPT_DIR}/
PYROCSHMEM_DIR=${SCRIPT_DIR}/../pyrocshmem
ROCSHMEM_ROOT=${SCRIPT_DIR}/../rocshmem_build/install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROCSHMEM_ROOT}/lib:${PYROCSHMEM_DIR}/../ompi_build/install/ompi/lib/

export PYTHONPATH=$PYTHONPATH:${PYROCSHMEM_DIR}/build:$TRITON_ROCSHMEM_DIR

export PYTHONPATH=$PYTHONPATH:${TRITON_ROCSHMEM_DIR}:${PYROCSHMEM_DIR}/build

export TRITON_CACHE_DIR=triton_cache
export ROCSHMEM_HOME=${ROCSHMEM_ROOT}
mkdir -p triton_cache

# run_ag_gemm
function run_rocshmem_sample() {
  pushd ${TRITON_ROCSHMEM_DIR}
  torchrun --nproc_per_node=8 --nnodes=1 example/sample.py
}

function run_rocm_ipc_handle() {
  pushd ${TRITON_ROCSHMEM_DIR}
  torchrun --nnodes 1 --nproc_per_node 4 example/test_torch_rocm_distributed.py
  popd
}
#run_rocshmem_sample
run_rocm_ipc_handle
