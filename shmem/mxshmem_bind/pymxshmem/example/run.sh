#!/bin/bash
MXSHMEM_LIB_PATH="$(pip show pymxshmem | grep "Location" | awk '{print $2}')/pymxshmem/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MXSHMEM_LIB_PATH}"
export MXSHMEM_DISABLE_CUDA_VMM=1
export MXSHMEM_BOOTSTRAP=UID
export MXSHMEM_IB_ENABLE_IBRC=0
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 run_ring_put.py
