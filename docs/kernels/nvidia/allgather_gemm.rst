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
   :param ctx: AllGatherGEMMTensorParallelContext containing communication state
   :returns: Output tensor of shape ``[M, N]``

.. py:function:: create_ag_gemm_context(max_M, N, K, dtype, rank, num_ranks, num_local_ranks=8, ag_intranode_stream=None, ag_internode_stream=None)

   Creates the context for AG-GEMM operation.

   :param max_M: Maximum M dimension
   :param N: N dimension (must be divisible by ``num_ranks``)
   :param K: K dimension
   :param dtype: Input/output data type
   :param rank: Current rank ID
   :param num_ranks: Total number of ranks
   :param num_local_ranks: Number of local ranks (default: ``8``)
   :param ag_intranode_stream: Optional CUDA stream for intra-node AllGather
   :param ag_internode_stream: Optional CUDA stream for inter-node AllGather
   :returns: AllGatherGEMMTensorParallelContext object

.. py:function:: gemm_persistent(...)

   Persistent GEMM kernel that consumes AllGather results.

.. py:function:: gemm_non_persistent(...)

   Non-persistent GEMM kernel variant.

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context

   # Create context from the tensor dimensions and dtype
   max_M = A.shape[0] * world_size
   ctx = create_ag_gemm_context(max_M, B.shape[0], A.shape[1], A.dtype, rank, world_size)

   # Perform AllGather + GEMM
   output = ag_gemm(A, B, ctx)

