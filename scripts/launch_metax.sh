#!/bin/bash
export MXSHMEM_BOOTSTRAP=MPI
export MXSHMEM_DISABLE_CUDA_VMM=1
export MXSHMEM_IB_ENABLE_IBGDA=1
export MXSHMEM_IB_ENABLE_IBRC=0

nproc_per_node=${ARNOLD_WORKER_GPU:=$(mx-smi --list | grep "GPU" | wc -l)}
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
