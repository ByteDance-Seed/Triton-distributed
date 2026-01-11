#!/bin/bash
nproc_per_node=1
nnodes=${WORKER_NUM:=1}
node_rank=${WORKER_ID:=0}

master_addr="127.0.0.1"
master_port="23459"

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
