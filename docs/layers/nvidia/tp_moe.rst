Tensor Parallel MoE
===================

Tensor Parallel Mixture of Experts layer for distributed inference.

Description
-----------

The TP MoE layer provides distributed MoE computation with
efficient expert routing and tensor parallelism.

Example Usage
-------------

.. code-block:: bash

   # Test TP MoE
   bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_moe.py \
       --bsz 32 --seq_len 128 --model <moe_model_path>

