Expert Parallelism All-to-All Layer
====================================

High-level layer API for Expert Parallelism All-to-All communication in MoE models.

Overview
--------

``EPAll2AllLayer`` provides a complete solution for token dispatch and combine operations
in Expert Parallelism, handling both intra-node and inter-node communication transparently.

Key Features
^^^^^^^^^^^^

- **Automatic Topology Detection**: Selects optimal kernels for single-node vs multi-node
- **Dynamic Buffer Management**: Automatically resizes output buffers based on actual token counts
- **Weight Transfer Support**: Optionally transfers routing weights along with tokens
- **Scatter Index Precomputation**: Supports external scatter index computation for advanced routing
- **AOT Compilation**: Optional ahead-of-time compilation for reduced JIT overhead

Architecture
------------

.. code-block:: text

   EPAll2AllLayer
   ├── EPConfig                    # Configuration parameters
   ├── DispatchCombineContext      # Symmetric memory buffers
   │   ├── token_send_buf_rdma     # [nnodes, max_tokens, hidden]
   │   ├── dispatch_output_buf     # [recv_tokens, hidden] (resizable)
   │   ├── topk_indices_buf_rdma   # [nnodes, max_tokens, topk]
   │   ├── weight_send/recv_buf    # Routing weights
   │   ├── signal_buf              # NVSHMEM signals
   │   └── ...
   └── BarrierAllContext           # Intra-node barrier

Workflow
--------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        dispatch() Flow                         │
   ├─────────────────────────────────────────────────────────────────┤
   │  1. Copy input to symmetric buffer                             │
   │  2. preprocess() - Compute routing metadata                    │
   │     ├── bincount(expert_indices)                              │
   │     ├── get_ag_splits_and_recv_offset_for_dispatch()          │
   │     └── [inter-node] get_dispatch_send_reqs()                 │
   │  3. init_output_buffer() - Poll CPU for buffer size           │
   │  4. dispatch_token() - Execute dispatch kernel                │
   │     ├── [inter-node] ep_dispatch_token_inplace()              │
   │     └── [intra-node] kernel_dispatch_token_intra_node()       │
   │  5. dispatch_postprocess() - Reset buffers                    │
   │  6. Return (output, weights, layout_desc)                     │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                        combine() Flow                          │
   ├─────────────────────────────────────────────────────────────────┤
   │  1. Copy expert output to symmetric buffer                     │
   │  2. combine_token_intra_node_and_send()                       │
   │     ├── [intra-node] kernel_combine_token_intra_node()        │
   │     └── [inter-node] ep_combine_token_inplace()               │
   │  3. [inter-node] Sum across nodes                             │
   │  4. Return combined output                                     │
   └─────────────────────────────────────────────────────────────────┘

API Reference
-------------

EPAll2AllLayer
^^^^^^^^^^^^^^

.. py:class:: EPAll2AllLayer(ep_group, max_tokens, hidden, topk, rank, num_tot_experts, local_world_size, world_size, dtype=torch.bfloat16, weight_dtype=torch.float32, num_sm=20, enable_local_combine=False, use_aot=False)

   High-level layer for EP All-to-All communication.

   :param ep_group: PyTorch distributed process group for EP
   :param max_tokens: Maximum number of tokens per rank
   :param hidden: Hidden dimension size
   :param topk: Number of experts selected per token
   :param rank: Current rank
   :param num_tot_experts: Total number of experts across all ranks
   :param local_world_size: Number of ranks per node (typically 8)
   :param world_size: Total number of ranks
   :param dtype: Token data type (default: ``torch.bfloat16``)
   :param weight_dtype: Routing weight data type (default: ``torch.float32``)
   :param num_sm: Number of SMs to use for kernels (default: 20)
   :param enable_local_combine: Enable intra-node local combine optimization (default: False)
   :param use_aot: Use AOT-compiled kernels (default: False)

   .. py:method:: dispatch(input, exp_indices, weight=None, full_scatter_indices=None)

      Dispatch tokens to their assigned experts.

      :param input: ``[num_tokens, hidden]`` - Input tokens
      :param exp_indices: ``[num_tokens, topk]`` - Expert indices from Top-K gate
      :param weight: ``[num_tokens, topk]`` - Optional routing weights (default: None)
      :param full_scatter_indices: ``[num_tokens, topk]`` - Optional precomputed scatter indices
      :returns: Tuple of (output, weights, layout_desc)

         - output: ``[recv_tokens, hidden]`` - Received tokens
         - weights: ``[recv_tokens]`` - Received weights (or None)
         - layout_desc: ``EPAllToAllLayoutDesc`` - Metadata for combine

   .. py:method:: combine(input, ep_a2a_layout_desc)

      Combine expert outputs back to original token positions.

      :param input: ``[recv_tokens, hidden]`` - Expert output tokens
      :param ep_a2a_layout_desc: Layout descriptor from dispatch
      :returns: ``[num_dispatch_tokens, hidden]`` - Combined output

   .. py:method:: finalize()

      Release NVSHMEM symmetric memory buffers.

   .. py:method:: ep_barrier_all(stream, intra_node_only=False)

      Synchronize all ranks (or just intra-node ranks).

      :param stream: CUDA stream
      :param intra_node_only: Only synchronize within node (default: False)

EPAllToAllLayoutDesc
^^^^^^^^^^^^^^^^^^^^

.. py:class:: EPAllToAllLayoutDesc

   Descriptor containing routing metadata from dispatch, needed for combine.

   .. py:attribute:: num_dispatch_token_cur_rank
      :type: int

      Number of tokens dispatched by this rank.

   .. py:attribute:: num_input_tokens_per_rank
      :type: torch.Tensor

      ``[world_size]`` - Number of tokens dispatched by each rank.

   .. py:attribute:: send_reqs_recv_tensor
      :type: Optional[torch.Tensor]

      ``[nnodes, 2, max_tokens]`` - Received send requests (inter-node only).

   .. py:attribute:: topk_indices_tensor
      :type: torch.Tensor

      ``[nnodes, max_tokens, topk]`` or ``[max_tokens, topk]`` - Expert indices.

   .. py:attribute:: token_dst_scatter_idx
      :type: torch.Tensor

      ``[nnodes, max_tokens, topk]`` - Scatter indices in output buffer.

   .. py:attribute:: skipped_token_mapping_indices
      :type: Optional[torch.Tensor]

      Intra-node optimization: maps skipped tokens to first occurrence.

   .. py:attribute:: skipped_token_topk_mapping_indices
      :type: Optional[torch.Tensor]

      Intra-node optimization: per-TopK mapping for skipped tokens.

EPConfig
^^^^^^^^

.. py:class:: EPConfig

   Configuration for EP All-to-All.

   .. py:attribute:: max_tokens
      :type: int

   .. py:attribute:: hidden
      :type: int

   .. py:attribute:: topk
      :type: int

   .. py:attribute:: num_experts
      :type: int

   .. py:attribute:: rank
      :type: int

   .. py:attribute:: world_size
      :type: int

   .. py:attribute:: local_world_size
      :type: int

   .. py:attribute:: token_dtype
      :type: torch.dtype

   .. py:attribute:: weight_dtype
      :type: torch.dtype

   .. py:attribute:: offset_dtype
      :type: torch.dtype

   .. py:property:: num_experts_per_rank
      :type: int

   .. py:property:: is_intra_node
      :type: bool

DispatchCombineContext
^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: DispatchCombineContext

   Manages symmetric memory buffers for dispatch and combine operations.

   .. py:method:: create(ep_config, capacity=2) -> DispatchCombineContext
      :classmethod:

      Create context with NVSHMEM symmetric buffers.

      :param ep_config: EPConfig instance
      :param capacity: Multiplier for output buffer capacity (default: 2)

   .. py:method:: finalize()

      Release all symmetric memory buffers.

   .. py:method:: reallocate_dispatch_output_buf(dispatch_recv_tokens)

      Resize output buffer if needed.

      :param dispatch_recv_tokens: Required capacity
      :returns: Tuple of (dispatch_output_buf, weight_recv_buf)

Usage Example
-------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch.distributed as dist
   from triton_dist.utils import initialize_distributed
   from triton_dist.layers.nvidia import EPAll2AllLayer

   # Initialize distributed runtime
   rank, world_size = initialize_distributed()

   # Create EP group (all ranks)
   ep_group = dist.group.WORLD

   # Create layer
   ep_layer = EPAll2AllLayer(
       ep_group=ep_group,
       max_tokens=256,
       hidden=7168,
       topk=8,
       rank=rank,
       num_tot_experts=256,      # 256 experts / 32 ranks = 8 per rank
       local_world_size=8,
       world_size=32,
       dtype=torch.bfloat16,
       weight_dtype=torch.float32,
       num_sm=20,
   )

   # Simulate input
   tokens = torch.randn(128, 7168, dtype=torch.bfloat16, device="cuda")
   expert_ids = torch.randint(0, 256, (128, 8), dtype=torch.int32, device="cuda")
   routing_weights = torch.softmax(torch.randn(128, 8, device="cuda"), dim=-1).float()

   # Dispatch
   recv_tokens, recv_weights, layout_desc = ep_layer.dispatch(
       input=tokens,
       exp_indices=expert_ids,
       weight=routing_weights,
   )

   # Expert FFN computation (example)
   # recv_tokens: [recv_count, hidden]
   expert_output = your_expert_ffn(recv_tokens)  

   # Combine
   combined = ep_layer.combine(expert_output, layout_desc)
   # combined: [128, 7168] - back to original token positions

   # Cleanup
   ep_layer.finalize()

With AOT Compilation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Enable AOT for production
   ep_layer = EPAll2AllLayer(
       ...,
       use_aot=True,  # Use pre-compiled kernels
   )

With Local Combine Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For intra-node scenarios with tokens routed to multiple same-rank experts
   ep_layer = EPAll2AllLayer(
       ...,
       world_size=8,             # Single node
       local_world_size=8,
       enable_local_combine=True, # Pre-aggregate locally
   )

Performance Tips
----------------

1. **Buffer Sizing**: Set ``max_tokens`` to your expected maximum to avoid reallocation
2. **AOT Compilation**: Enable ``use_aot=True`` in production for faster startup
3. **SM Count**: Tune ``num_sm`` based on your model's compute requirements
4. **Local Combine**: Enable for intra-node when TopK > 1 and experts overlap

.. note::

   The layer automatically selects between intra-node and inter-node kernels 
   based on ``world_size`` and ``local_world_size``.

See Also
--------

- :doc:`/kernels/nvidia/ep_a2a` - Underlying dispatch/combine kernels
- :doc:`/kernels/nvidia/ep_a2a_intra_node` - Intra-node optimized kernels
- :doc:`ep_ll_a2a_layer` - Low-latency variant with FP8 quantization
