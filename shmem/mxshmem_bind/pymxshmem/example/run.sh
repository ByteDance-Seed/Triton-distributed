#!/bin/bash
export MXSHMEM_BOOTSTRAP=UID
export MXSHMEM_IB_ENABLE_IBRC=0
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 run_ring_put.py
