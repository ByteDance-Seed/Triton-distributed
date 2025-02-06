#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
TRITON_ROCSHMEM_DIR=${SCRIPT_DIR}/../rocshmem_bind/python
PYROCSHMEM_DIR=${SCRIPT_DIR}/../rocshmem_bind/pyrocshmem
ROCSHMEM_ROOT=${SCRIPT_DIR}/../rocshmem_bind/rocshmem_build/install
MPI_ROOT=${SCRIPT_DIR}/../rocshmem_bind/ompi_build/install/ompi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROCSHMEM_ROOT}/lib:${MPI_ROOT}/lib
echo $LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:${PYROCSHMEM_DIR}/build:$TRITON_ROCSHMEM_DIR
export PYTHONPATH=$PYTHONPATH:${TRITON_ROCSHMEM_DIR}:${PYROCSHMEM_DIR}/build
export TRITON_CACHE_DIR=triton_cache
export ROCSHMEM_HOME=${ROCSHMEM_ROOT}
## AMD env vars
export TRITON_HIP_USE_BLOCK_PINGPONG=1 # for gemm perf
export GPU_STREAMOPS_CP_WAIT=1
export DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1
# export AMD_LOG_LEVEL=5 # for debug
mkdir -p triton_cache
nnodes=1
node_rank=0
nproc_per_node=8
CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  $@"
echo ${CMD}
${CMD}
