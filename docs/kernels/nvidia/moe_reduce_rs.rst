MoE ReduceScatter
=================

MoE (Mixture of Experts) ReduceScatter kernel for tensor parallelism.

API Reference
-------------

.. py:function:: create_moe_rs_context(...)

   Creates context for MoE ReduceScatter operation.

Example Usage
-------------

.. code-block:: bash

   # Test MoE ReduceScatter
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2

