#!/bin/bash
# Launch FlashComm multi-GPU dispatch test through the locked uv environment.
# Usage:
#   bash tools/nv/sm90/launch_flashcomm_test.sh          # auto: 2, 4, 8
#   bash tools/nv/sm90/launch_flashcomm_test.sh 2        # 2 GPUs only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEST_SCRIPT="${PROJECT_ROOT}/tests/nv/sm90/distributed/test_flashcomm_multi_gpu.py"

cd "$PROJECT_ROOT"

if [ -z "$1" ]; then
    echo "========================================================================"
    echo "FlashComm Multi-GPU Dispatch Test (auto: 2, 4, 8)"
    echo "========================================================================"
    uv run --locked python "$TEST_SCRIPT"
else
    echo "========================================================================"
    echo "FlashComm Multi-GPU Dispatch Test: $1 GPUs"
    echo "========================================================================"
    uv run --locked python "$TEST_SCRIPT" --num-ranks "$1"
fi
