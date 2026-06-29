#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
pushd "${PROJECT_ROOT}"

export PYTHONPATH=${PYTHONPATH:-}:$(realpath python)

TMA_DIR="python/triton_dist/test/nvidia/tma"

# single-GPU tests
python3 ${TMA_DIR}/test_smem_ops.py
python3 ${TMA_DIR}/test_tma_foundation.py
python3 ${TMA_DIR}/test_tma_mixed_simt.py
python3 ${TMA_DIR}/test_tma_pipeline.py

# multi-GPU A2A tests
bash scripts/launch.sh ${TMA_DIR}/test_post_attn_a2a_op.py --verify --q_nheads 64 --hd 128 --max_seq 16384 --iters 20
bash scripts/launch.sh ${TMA_DIR}/test_pre_attn_a2a_op.py --verify --q_nheads 64 --hd 128 --max_seq 16384 --iters 20
bash scripts/launch.sh ${TMA_DIR}/test_pre_attn_a2a_op.py --verify --qkv_pack --q_nheads 64 --k_nheads 8 --v_nheads 8 --hd 128 --max_seq 16384 --iters 20
