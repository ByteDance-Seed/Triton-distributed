#!/bin/bash
# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

bash ./launch_single_device.sh ./test_distributed_wait.py --case correctness
bash ./launch.sh ./test_ag_gemm_intra_node.py --case correctness_no_tma
python test_common_ops.py
