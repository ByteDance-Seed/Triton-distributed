AllGather GEMM
==============

The AllGather GEMM kernel fuses AllGather collective communication with GEMM computation,
enabling computation-communication overlapping.

API Reference
-------------

.. py:function:: ag_gemm(a, b, ctx, ...)

   Performs AllGather followed by GEMM with overlapping.

   :param a: Local input tensor of shape ``[M_per_rank, K]``
   :param b: Weight tensor of shape ``[N, K]``
   :param ctx: AGGemmContext containing symmetric memory and signals
   :returns: Output tensor of shape ``[M, N]``

.. py:function:: create_ag_gemm_context(local_tensor, weight, rank, num_ranks, max_M, BLOCK_M, BLOCK_N, BLOCK_K, stages)

   Creates the context for AG-GEMM operation.

   :param local_tensor: Sample local tensor for shape inference
   :param weight: Weight tensor
   :param rank: Current rank ID
   :param num_ranks: Total number of ranks
   :param max_M: Maximum M dimension
   :param BLOCK_M: Block size in M dimension
   :param BLOCK_N: Block size in N dimension
   :param BLOCK_K: Block size in K dimension
   :param stages: Number of pipeline stages
   :returns: AGGemmContext object

.. py:function:: gemm_persistent(...)

   Persistent GEMM kernel that consumes AllGather results.

.. py:function:: gemm_non_persistent(...)

   Non-persistent GEMM kernel variant.

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context

   # Create context
   ctx = create_ag_gemm_context(A, B, rank, world_size, max_M=M, 
                                 BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3)

   # Perform AllGather + GEMM
   output = ag_gemm(A, B, ctx)

