#!/bin/bash
export NCCL_DEBUG=WARN
export ROCSHMEM_BACKEND=GDA
export ROCSHMEM_GDA_PROVIDER=mlx5
HIP_VISIBLE_DEVICES=0,1 ROCM_VISIBLE_DEVICES=4,5,6,7 ARNOLD_WORKER_GPU=2 ARNOLD_WORKER_NUM=2 ARNOLD_ID=0 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568 &
HIP_VISIBLE_DEVICES=2,3 ROCM_VISIBLE_DEVICES=4,5,6,7 ARNOLD_WORKER_GPU=2 ARNOLD_WORKER_NUM=2 ARNOLD_ID=1 bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568 &
wait $(jobs -p)