GEMM ReduceScatter (AMD)
========================

GEMM + ReduceScatter kernel for AMD GPUs with intra-node computation-communication overlapping.

API Reference
-------------

.. py:function:: gemm_rs_intra_node(a, b, ctx, ...)

   Performs GEMM followed by ReduceScatter with overlapping on AMD GPUs.

   :param a: Input tensor of shape ``[M, K]``
   :param b: Weight tensor of shape ``[N, K]``
   :param ctx: GemmRSIntraNodeContext
   :returns: Output tensor of shape ``[M/world_size, N]``

.. py:function:: create_gemm_rs_intra_node_context(max_M, N, output_dtype, rank, world_size, tp_group, fuse_scatter=True)

   Creates the context for GEMM-RS intra-node operation on AMD GPUs.

   :param max_M: Maximum M dimension
   :param N: N dimension
   :param output_dtype: Output data type
   :param rank: Current rank ID
   :param world_size: Total number of ranks
   :param tp_group: Tensor parallel process group
   :param fuse_scatter: Whether to fuse scatter into GEMM
   :returns: GemmRSIntraNodeContext object

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.amd import gemm_rs_intra_node, create_gemm_rs_intra_node_context

   # Create context
   ctx = create_gemm_rs_intra_node_context(M, N, torch.float16,
                                            rank, world_size, tp_group)

   # Perform GEMM + ReduceScatter
   output = gemm_rs_intra_node(input, weight, ctx)

