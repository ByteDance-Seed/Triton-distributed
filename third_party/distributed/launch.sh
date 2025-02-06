#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-1000000000}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DISTRIBUTED_DIR=$SCRIPT_DIR
THIRD_PARTY_DIR=$(dirname -- "$SCRIPT_DIR")

NVSHMEM_ROOT=${THIRD_PARTY_DIR}/nvshmem/build/install

export LD_LIBRARY_PATH=${NVSHMEM_ROOT}/lib:$LD_LIBRARY_PATH
export NVSHMEM_DISABLE_CUDA_VMM=${NVSHMEM_DISABLE_CUDA_VMM:-1} # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export TRITON_CACHE_DIR=$DISTRIBUTED_DIR/.triton

mkdir -p $DISTRIBUTED_DIR/.triton

nproc_per_node=$(nvidia-smi --list-gpus | wc -l)
nnodes=${WORKER_NUM:=1}
node_rank=${WORKER_ID:=0}

master_addr="127.0.0.1"
master_port="23456"

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
