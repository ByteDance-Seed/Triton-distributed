GEMM AllReduce
==============

GEMM + AllReduce fused kernels for efficient tensor parallelism.

API Reference
-------------

.. py:function:: gemm_allreduce_op(a, b, ctx, ...)

   Performs GEMM followed by AllReduce with overlapping.

.. py:function:: create_gemm_ar_context(...)

   Creates context for GEMM + AllReduce.

.. py:function:: low_latency_gemm_allreduce_op(a, b, ctx, ...)

   Low-latency GEMM + AllReduce for small batch sizes.

.. py:function:: create_ll_gemm_ar_context(...)

   Creates context for low-latency GEMM + AllReduce.

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia import (
       gemm_allreduce_op, 
       create_gemm_ar_context
   )

   # Create context
   ctx = create_gemm_ar_context(...)

   # Perform GEMM + AllReduce
   output = gemm_allreduce_op(input, weight, ctx)

