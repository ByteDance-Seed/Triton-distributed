Ulysses Sequence Parallelism
============================

Ulysses-style Sequence Parallelism communication kernels.

API Reference
-------------

.. py:function:: create_ulysses_sp_pre_attn_comm_context(...)

   Creates context for Ulysses SP pre-attention communication.

All-to-All Single GEMM
----------------------

.. py:function:: create_all_to_all_single_gemm_context(...)

   Creates context for All-to-All + GEMM fusion.

.. py:function:: all_to_all_single_gemm(...)

   Fused All-to-All + GEMM operation.

Example Usage
-------------

.. code-block:: bash

   # Test Ulysses SP Dispatch
   bash scripts/launch.sh python/triton_dist/test/nvidia/test_ulysses_sp_dispatch.py 1 8000 32 128 --gqa 8 --check

