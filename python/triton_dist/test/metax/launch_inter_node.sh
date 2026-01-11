#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export MCCL_DEBUG=ERROR

export MXSHMEM_SYMMETRIC_SIZE=${MXSHMEM_SYMMETRIC_SIZE:-1000000000}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# internode
export MXSHMEM_BOOTSTRAP=MPI
export MXSHMEM_DISABLE_CUDA_VMM=1
export MXSHMEM_IB_ENABLE_IBGDA=1
export MXSHMEM_IB_ENABLE_IBRC=0
# SET SOCKET NAME
export MXSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=ens108np0   #en,eth0,em,bond
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:~/.triton}

nproc_per_node=$(mx-smi --list | grep "GPU" | wc -l)
nnodes=${WORKER_NUM:=1}
node_rank=${WORKER_ID:=0}

master_addr="10.2.179.27"
master_port="23468"

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${DIST_TRITON_EXTRA_TORCHRUN_ARGS} \
  ${additional_args} \
  ${DIST_TRITON_EXTRA_TORCHRUN_ARGS} \
  $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
