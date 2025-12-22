Qwen MoE Model
==============

Qwen MoE (Mixture of Experts) model implementation for distributed inference.

Description
-----------

The Qwen MoE module provides a complete implementation of Qwen MoE models
with tensor parallelism and expert parallelism support.

Example Usage
-------------

.. code-block:: bash

   # Test MoE E2E
   bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_e2e_inference.py \
       --bsz 4096 --gen_len 128 --max_length 150 --model <moe_model_path> --backend triton_dist

