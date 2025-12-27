Tensor Parallel Attention
=========================

Tensor Parallel Attention layer for distributed inference.

Description
-----------

The TP Attention layer provides distributed attention computation with
AllGather/ReduceScatter overlapping for efficient tensor parallelism.

Modes
-----

- ``ag_rs``: AllGather input + ReduceScatter output
- ``allreduce``: AllReduce-based parallelism
- ``gemm_ar``: GEMM + AllReduce fusion

Example Usage
-------------

.. code-block:: bash

   # Prefill mode
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py \
       --bsz 32 --seq_len 128 --model <model_path> --run_type prefill --mode ag_rs

   # Decode mode
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py \
       --bsz 128 --seq_len 128 --model <model_path> --run_type decode --mode ag_rs

