AllGather GEMM (AMD)
====================

AllGather + GEMM kernel for AMD GPUs with intra-node computation-communication overlapping.

API Reference
-------------

.. py:function:: ag_gemm_intra_node(a, b, ctx, ...)

   Performs AllGather followed by GEMM with overlapping on AMD GPUs.

   :param a: Local input tensor of shape ``[M_per_rank, K]``
   :param b: Weight tensor of shape ``[N, K]``
   :param ctx: AGGemmIntraNodeContext
   :returns: Output tensor of shape ``[M, N]``

.. py:function:: create_ag_gemm_intra_node_context(max_M, N, K, input_dtype, output_dtype, rank, world_size, tp_group, M_PER_CHUNK=256)

   Creates the context for AG-GEMM intra-node operation on AMD GPUs.

   :param max_M: Maximum M dimension
   :param N: N dimension
   :param K: K dimension
   :param input_dtype: Input data type
   :param output_dtype: Output data type
   :param rank: Current rank ID
   :param world_size: Total number of ranks
   :param tp_group: Tensor parallel process group
   :param M_PER_CHUNK: Chunk size for overlapping
   :returns: AGGemmIntraNodeContext object

Example Usage
-------------

.. code-block:: python

   from triton_dist.kernels.amd import ag_gemm_intra_node, create_ag_gemm_intra_node_context

   # Create context
   ctx = create_ag_gemm_intra_node_context(M, N, K, torch.float16, torch.float16,
                                            rank, world_size, tp_group)

   # Perform AllGather + GEMM
   output = ag_gemm_intra_node(input, weight, ctx)

Running the Test
----------------

.. code-block:: bash

   bash scripts/launch_amd.sh python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568

