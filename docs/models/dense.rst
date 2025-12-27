Dense Model
===========

Dense transformer model implementation for distributed inference.

Description
-----------

The dense model module provides a complete implementation of dense transformer models
(e.g., Qwen, LLaMA) with tensor parallelism support.

Features
--------

- Tensor Parallel attention and MLP
- Multiple parallelism modes (ag_rs, allreduce, gemm_ar)
- KV cache support for efficient decoding

Example Usage
-------------

.. code-block:: bash

   # End-to-end inference test
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py \
       --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist

