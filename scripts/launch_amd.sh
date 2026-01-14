#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
DISTRIBUTED_DIR=$(dirname -- "$SCRIPT_DIR")
TRITON_ROCSHMEM_DIR=${SCRIPT_DIR}/../shmem/rocshmem_bind/python
PYROCSHMEM_DIR=${SCRIPT_DIR}/../shmem/rocshmem_bind/pyrocshmem
ROCSHMEM_ROOT=${SCRIPT_DIR}/../shmem/rocshmem_bind/rocshmem_build/install
MPI_ROOT="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROCSHMEM_ROOT}/lib:${MPI_ROOT}/lib

case ":${PYTHONPATH}:" in
    *:"${DISTRIBUTED_DIR}/python:${PYROCSHMEM_DIR}/build:${TRITON_ROCSHMEM_DIR}":*)
        ;;
    *)
        export PYTHONPATH="${PYTHONPATH}:${DISTRIBUTED_DIR}/python:${PYROCSHMEM_DIR}/build:${TRITON_ROCSHMEM_DIR}"
        ;;
esac

export TRITON_CACHE_DIR=triton_cache
export ROCSHMEM_HOME=${ROCSHMEM_ROOT}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:=IPC}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:=mlx5} # Only used with backend GDA

## AMD env vars
export TRITON_HIP_USE_BLOCK_PINGPONG=1 # for gemm perf
export GPU_STREAMOPS_CP_WAIT=1
export DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1
# export AMD_LOG_LEVEL=5 # for debug

mkdir -p triton_cache

nproc_per_node=${ARNOLD_WORKER_GPU:=$(rocm-smi | grep W | wc -l)}
nnodes=${ARNOLD_WORKER_NUM:=1}
node_rank=${ARNOLD_ID:=0}

master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="23457"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args} \
  ${DIST_TRITON_EXTRA_TORCHRUN_ARGS} \
  $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
