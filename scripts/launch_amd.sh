#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
DISTRIBUTED_DIR=$(dirname -- "$SCRIPT_DIR")
TRITON_ROCSHMEM_DIR=${SCRIPT_DIR}/../shmem/rocshmem_bind/python
PYROCSHMEM_DIR=${SCRIPT_DIR}/../shmem/rocshmem_bind/pyrocshmem
ROCSHMEM_ROOT=${SCRIPT_DIR}/../shmem/rocshmem_bind/rocshmem_build/install
MPI_ROOT="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"

# Only add rocshmem and MPI to LD_LIBRARY_PATH if not using mori_shmem backend
if [ "${TRITON_DIST_SHMEM_BACKEND}" != "mori_shmem" ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROCSHMEM_ROOT}/lib:${MPI_ROOT}/lib
fi

case ":${PYTHONPATH}:" in
    *:"${DISTRIBUTED_DIR}/python:${PYROCSHMEM_DIR}/build:${TRITON_ROCSHMEM_DIR}":*)
        ;;
    *)
        export PYTHONPATH="${PYTHONPATH}:${DISTRIBUTED_DIR}/python:${PYROCSHMEM_DIR}/build:${TRITON_ROCSHMEM_DIR}"
        ;;
esac

# Export TRITON_PLUGIN_PATHS before any child imports triton: the out-of-tree
# distributed dialects/passes/ops live in libtriton_dist.so, which Triton loads
# exactly once at first `import triton`. Resolving it via triton_dist._plugin
# does not import triton, so it is safe here. Mirrors the NVIDIA
# scripts/setenv.sh::set_triton_dist_plugin (sourced by scripts/launch.sh).
if [ -z "${TRITON_PLUGIN_PATHS}" ]; then
    _dist_plugin="$(python3 -c 'import triton_dist._plugin as p; print(p.find_plugin() or "")' 2>/dev/null)"
    if [ -n "${_dist_plugin}" ] && [ -f "${_dist_plugin}" ]; then
        export TRITON_PLUGIN_PATHS="${_dist_plugin}"
        echo "TRITON_PLUGIN_PATHS=${TRITON_PLUGIN_PATHS}"
    else
        echo "WARNING: libtriton_dist.so not found; build it via 'pip install ./python'"
    fi
fi

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-triton_cache}
export ROCSHMEM_HOME=${ROCSHMEM_ROOT}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:=IPC}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:=mlx5} # Only used with backend GDA

## AMD env vars
export TRITON_HIP_USE_BLOCK_PINGPONG=1 # for gemm perf
export GPU_STREAMOPS_CP_WAIT=1
export DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1
# export AMD_LOG_LEVEL=5 # for debug

mkdir -p ${TRITON_CACHE_DIR}

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
# NOTE(local): mori_shmem's C++ layer reads MASTER_ADDR/MASTER_PORT directly from
# the environment (it aborts with "requires torchrun/torch.distributed env vars"
# otherwise). torch.distributed.run's rendezvous path does NOT export those, so
# set them explicitly for the mori backend.
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"
# NOTE(local): use `python3 -m torch.distributed.run` instead of the `torchrun`
# console script. torchrun's shebang pins /usr/local/bin/python (system), so its
# rank subprocesses would run under the system interpreter and miss the venv's
# hip-python / triton / triton_dist (ModuleNotFoundError: hip). Going through the
# active python3 keeps every rank inside the venv.
CMD="python3 -m torch.distributed.run \
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
