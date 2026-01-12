#!/bin/bash
bash ./launch_single_device.sh ./test_distributed_wait.py --case correctness
bash ./launch.sh ./test_ag_gemm_intra_node.py --case correctness_no_tma
python test_common_ops.py
bash ./launch_inter_node.sh ./test_ag_gemm_inter_node.py
