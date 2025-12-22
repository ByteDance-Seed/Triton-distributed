Tensor Parallel MLP
===================

Tensor Parallel MLP layer for distributed inference.

Description
-----------

The TP MLP layer provides distributed MLP computation with
computation-communication overlapping for efficient tensor parallelism.

Modes
-----

- ``ag_rs``: AllGather input + ReduceScatter output
- ``allreduce``: AllReduce-based parallelism  
- ``gemm_ar``: GEMM + AllReduce fusion

Example Usage
-------------

.. code-block:: bash

   # Test TP MLP
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py \
       --M 4096 --model <model_path> --mode ag_rs

