#!/bin/bash
# Launch FlashComm multi-GPU dispatch test.
# Usage:
#   bash design/launch_flashcomm_test.sh          # auto: 2, 4, 8
#   bash design/launch_flashcomm_test.sh 2        # 2 GPUs only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_flashcomm_multi_gpu.py"

if [ -z "$1" ]; then
    echo "========================================================================"
    echo "FlashComm Multi-GPU Dispatch Test (auto: 2, 4, 8)"
    echo "========================================================================"
    python3 "$TEST_SCRIPT"
else
    echo "========================================================================"
    echo "FlashComm Multi-GPU Dispatch Test: $1 GPUs"
    echo "========================================================================"
    python3 "$TEST_SCRIPT" --num-ranks "$1"
fi
