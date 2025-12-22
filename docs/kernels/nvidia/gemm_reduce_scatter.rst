GEMM ReduceScatter
==================

The GEMM ReduceScatter kernel fuses GEMM computation with ReduceScatter collective communication,
enabling computation-communication overlapping.

API Reference
-------------

.. py:function:: gemm_rs(a, b, ctx, ...)

   Performs GEMM followed by ReduceScatter with overlapping.

   :param a: Input tensor of shape ``[M, K]``
   :param b: Weight tensor of shape ``[N, K]``
   :param ctx: GemmRSContext containing symmetric memory and signals
   :returns: Output tensor of shape ``[M/world_size, N]``

.. py:function:: create_gemm_rs_context(max_M, N, rank, world_size, local_world_size, output_dtype, rs_stream=None)

   Creates the context for GEMM-RS operation.

   :param max_M: Maximum M dimension
   :param N: N dimension
   :param rank: Current rank ID
   :param world_size: Total number of ranks
   :param local_world_size: Number of ranks per node
   :param output_dtype: Output data type
   :param rs_stream: Optional CUDA stream for ReduceScatter
   :returns: GemmRSContext object

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia import gemm_rs, create_gemm_rs_context

   # Create context
   ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size, 
                                  output_dtype=torch.bfloat16)

   # Perform GEMM + ReduceScatter
   output = gemm_rs(input, weight, ctx)

