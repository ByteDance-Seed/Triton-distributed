Expert Parallelism All-to-All (EP A2A)
======================================

Expert Parallelism (EP) All-to-All communication kernels for Mixture-of-Experts (MoE) models.
These kernels enable efficient token dispatch and combine operations across distributed experts.

Overview
--------

In MoE models with Expert Parallelism, tokens need to be routed to their assigned experts 
which may reside on different ranks. The EP A2A kernels provide:

1. **Dispatch**: Scatter input tokens to remote experts based on Top-K routing decisions
2. **Combine**: Gather expert outputs back to the original ranks and perform weighted summation

.. code-block:: text

   Dispatch Phase:
   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
   │   Rank 0       │    │   Rank 1       │    │   Rank N       │
   │ [tokens]       │    │ [tokens]       │    │ [tokens]       │
   │   ↓ TopK Gate  │    │   ↓ TopK Gate  │    │   ↓ TopK Gate  │
   │ expert_ids     │    │ expert_ids     │    │ expert_ids     │
   └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ↓
                    ┌───────────────────────┐
                    │   All-to-All Dispatch │
                    │   (Token Shuffle)     │
                    └───────────┬───────────┘
                                ↓
   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
   │ Expert 0..E/N  │    │Expert E/N..2E/N│    │Expert (N-1)E/N │
   │ [recv tokens]  │    │ [recv tokens]  │    │ [recv tokens]  │
   └────────────────┘    └────────────────┘    └────────────────┘

   Combine Phase (reverse direction):
   Expert outputs → All-to-All → Weighted sum per original token

Key Concepts
------------

Token Layout
^^^^^^^^^^^^

The kernel uses a hierarchical layout optimized for multi-node communication:

- **Inter-node buffer**: ``[nnodes, max_tokens, hidden]`` - tokens organized by source node
- **Intra-node buffer**: Symmetric memory for direct GPU-to-GPU access within a node
- **recv_buf_offset_per_expert**: ``[world_size, experts_per_rank, world_size]`` - 
  tracks where tokens from each source rank go for each local expert

Dispatch Workflow
^^^^^^^^^^^^^^^^^

1. **Preprocessing** (``get_ag_splits_and_recv_offset_for_dispatch``):
   
   - Count tokens per expert (bincount)
   - AllGather split information across all ranks
   - Compute destination offsets for each token
   - Generate send requests for inter-node transfers

2. **Token Dispatch** (``kernel_dispatch_token``):

   - Ring-based inter-node communication (reduces network congestion)
   - Use NVSHMEM ``putmem_nbi_warp`` for asynchronous data transfer
   - Signal-based synchronization between nodes
   - Optionally transfer routing weights

3. **Postprocessing**:

   - Reset symmetric buffers for next iteration
   - Handle skipped token mapping for intra-node optimization

Combine Workflow
^^^^^^^^^^^^^^^^

1. **Intra-node Reduce** (``kernel_combine_token``):

   - Load expert outputs from symmetric memory of peer ranks
   - Accumulate outputs for tokens routed to multiple local experts
   - Transfer reduced results to inter-node buffer

2. **Inter-node Reduce**:

   - Sum outputs across nodes for each original token

API Reference
-------------

Core Functions
^^^^^^^^^^^^^^

.. py:function:: ep_dispatch_token_inplace(send_reqs_for_nodes, signal_buf, recv_buf_offset_per_expert, send_buf, output_buf, weight_send_buf, weight_recv_buf, topk_indices_tensor, token_dst_scatter_idx, num_input_tokens_per_rank, max_tokens, topk, hidden, bytes_per_token, experts_per_rank, local_world_size, has_weight, with_scatter_indices, num_sms, use_aot=False)

   Dispatch tokens to their assigned experts across all ranks.

   :param send_reqs_for_nodes: ``[nnodes, 2, max_tokens]`` - Send request ranges per target node
   :param signal_buf: ``[world_size]`` - NVSHMEM signal buffer for synchronization
   :param recv_buf_offset_per_expert: ``[world_size, experts_per_rank, world_size]`` - Destination offsets
   :param send_buf: ``[nnodes, max_tokens, hidden]`` - Source token buffer (symmetric)
   :param output_buf: ``[recv_tokens, hidden]`` - Destination buffer for received tokens (symmetric)
   :param weight_send_buf: ``[nnodes, max_tokens, topk]`` - Optional routing weights to send
   :param weight_recv_buf: ``[recv_tokens]`` - Optional buffer for received weights
   :param topk_indices_tensor: ``[nnodes, max_tokens, topk]`` - Expert indices per token
   :param token_dst_scatter_idx: ``[nnodes, max_tokens, topk]`` - Scatter indices in output buffer
   :param num_input_tokens_per_rank: ``[world_size]`` - Token count per source rank
   :param max_tokens: Maximum tokens per rank
   :param topk: Number of experts per token
   :param hidden: Hidden dimension size
   :param bytes_per_token: ``hidden * dtype.itemsize``
   :param experts_per_rank: Number of experts per rank
   :param local_world_size: Number of ranks per node
   :param has_weight: Whether to transfer routing weights
   :param with_scatter_indices: Whether scatter indices are precomputed
   :param num_sms: Number of SMs to use
   :param use_aot: Whether to use AOT-compiled kernel

.. py:function:: ep_combine_token_inplace(counter_workspace, num_input_tokens_per_rank, send_reqs_recv_tensor, intra_node_reduce_buf, input, send_buf, topk_indices_tensor, token_dst_scatter_idx, max_tokens, topk, hidden, bytes_per_token, experts_per_rank, local_world_size, num_sms, use_aot=False)

   Combine expert outputs back to original token positions.

   :param counter_workspace: ``[nnodes]`` - Grid synchronization counters
   :param num_input_tokens_per_rank: ``[world_size]`` - Token count per source rank
   :param send_reqs_recv_tensor: ``[nnodes, 2, max_tokens]`` - Received send requests
   :param intra_node_reduce_buf: ``[nnodes, max_tokens, hidden]`` - Intermediate reduction buffer
   :param input: ``[recv_tokens, hidden]`` - Expert output tokens (symmetric)
   :param send_buf: ``[nnodes, max_tokens, hidden]`` - Output buffer for combined tokens
   :param topk_indices_tensor: ``[nnodes, max_tokens, topk]`` - Expert indices per token
   :param token_dst_scatter_idx: ``[nnodes, max_tokens, topk]`` - Scatter indices from dispatch
   :param max_tokens: Maximum tokens per rank
   :param topk: Number of experts per token
   :param hidden: Hidden dimension size
   :param bytes_per_token: ``hidden * dtype.itemsize``
   :param experts_per_rank: Number of experts per rank
   :param local_world_size: Number of ranks per node
   :param num_sms: Number of SMs to use
   :param use_aot: Whether to use AOT-compiled kernel

Preprocessing Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: get_ag_splits_and_recv_offset_for_dispatch(send_reqs_for_nodes, send_reqs_recv_bufs, exp_indices, topk_indices_buf, expert_indices_signal_buf, local_splits, full_splits_buf, splits_signal_buf, topk, local_world_size, world_size, max_tokens, experts_per_rank, full_scatter_indices=None, cpu_default_val=-1, offset_dtype=torch.int32, num_sm=20, use_aot=False)

   Compute routing metadata for dispatch operation.

   :returns: Tuple of (recv_buf_offset_per_expert, num_recv_tokens_per_rank_cpu, num_input_tokens_per_rank, token_dst_scatter_idx, send_reqs_recv_bufs_copy, topk_indices_buf_copy)

.. py:function:: get_dispatch_send_reqs(exp_indices, send_reqs_for_nodes, experts_per_rank, local_world_size, num_sms, use_aot=False)

   Generate send request ranges for inter-node dispatch.

   :param exp_indices: ``[num_tokens, topk]`` - Expert indices from Top-K gate
   :param send_reqs_for_nodes: ``[nnodes, 2, max_tokens]`` - Output buffer for (start, end) ranges

.. py:function:: bincount(input, length, output=None, output_dtype=torch.int32, num_sm=16, use_aot=False)

   Count tokens per expert (GPU-accelerated bincount).

   :param input: ``[num_tokens * topk]`` - Flattened expert indices
   :param length: Number of experts + 1 (for dropped tokens)
   :returns: Token counts per expert

Kernel Implementation Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``kernel_dispatch_token``
"""""""""""""""""""""""""

.. code-block:: python

   @triton_dist.jit
   def kernel_dispatch_token(
       send_reqs_for_nodes, signals_for_nodes, recv_buf_offset_per_expert,
       input_buf, output_buf, weight_send_buf, weight_recv_buf,
       topk_indices_tensor, token_dst_scatter_idx, num_input_tokens_per_rank,
       max_tokens, topk, hidden_size, bytes_per_token, num_sms,
       experts_per_rank: tl.constexpr, local_world_size: tl.constexpr,
       HAS_WEIGHT: tl.constexpr, WITH_SCATTER_INDICES: tl.constexpr,
   ):
       ...

Key algorithmic steps:

1. **Ring Communication**: Iterate through nodes in ring order to balance network load

   .. code-block:: python

      for node_offset in range(0, nnodes):
          target_node = (node_id + node_offset + 1) % nnodes

2. **Warp-level Data Transfer**: Each warp handles one token's data transfer

   .. code-block:: python

      libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)

3. **Atomic Offset Allocation**: Thread-safe destination slot allocation

   .. code-block:: python

      store_idx = atomic_add_per_warp(
          recv_buf_offset_per_expert + expert_rank * ..., 1, 
          scope="gpu", semantic="relaxed")

``kernel_combine_token``
""""""""""""""""""""""""

Key algorithmic steps:

1. **Remote Memory Access**: Read expert outputs directly from peer ranks

   .. code-block:: python

      remote_input_ptr = dl.symm_at(input_buf, expert_rank)
      token = dl.ld_vector(remote_input_ptr + offset, vec_size=vec_size)

2. **Vectorized Accumulation**: Use 128-bit vector operations for efficiency

   .. code-block:: python

      token_accum = dl.zeros_vector(vec_size, tl.float32)
      for j in range(topk):
          token = dl.ld_vector(...).to(tl.float32)
          token_accum = token_accum + token
      dl.st_vector(send_buf + ..., token_accum.to(send_buf.dtype.element_ty))

AOT Compilation
^^^^^^^^^^^^^^^

The kernels support Ahead-of-Time (AOT) compilation for reduced JIT overhead:

.. code-block:: python

   @aot_compile_spaces({
       "kernel_dispatch_token_bf16_weight_fp32": {
           "signature": kernel_dispatch_token_signature.format(...),
           "grid": ["num_sms", "1", "1"],
           "triton_algo_infos": [
               {"experts_per_rank": 64, "local_world_size": 8, ...},
               {"experts_per_rank": 32, ...},
               ...
           ]
       }
   })
   @triton_dist.jit
   def kernel_dispatch_token(...):
       ...

Performance Characteristics
---------------------------

- **Warp-level Parallelism**: Each warp handles one token, maximizing parallelism
- **Ring Communication**: Balances network load across nodes
- **Symmetric Memory**: Enables direct GPU-to-GPU access without CPU involvement
- **Vectorized Operations**: 128-bit vector loads/stores for maximum memory bandwidth
- **Grid Synchronization**: Efficient barrier implementation using atomic operations

.. note::

   For optimal performance:
   
   - Use ``num_sms=20`` or higher for large token counts
   - Enable AOT compilation (``use_aot=True``) in production
   - Ensure ``hidden`` is divisible by 16 for vectorization

Usage Example
-------------

.. code-block:: python

   from triton_dist.layers.nvidia import EPAll2AllLayer

   # Create EP layer
   ep_layer = EPAll2AllLayer(
       ep_group=ep_group,
       max_tokens=256,
       hidden=7168,
       topk=8,
       rank=rank,
       num_tot_experts=256,
       local_world_size=8,
       world_size=32,
       dtype=torch.bfloat16,
       weight_dtype=torch.float32,
   )

   # Dispatch tokens to experts
   output, weights, layout_desc = ep_layer.dispatch(
       input=tokens,           # [num_tokens, hidden]
       exp_indices=expert_ids, # [num_tokens, topk]
       weight=routing_weights, # [num_tokens, topk]
   )

   # After expert computation...
   combined = ep_layer.combine(expert_output, layout_desc)

   # Cleanup
   ep_layer.finalize()

Run Example
-----------

.. code-block:: sh
   
   NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check


See Also
--------

- :doc:`ep_a2a_intra_node` - Optimized kernels for single-node EP
- :doc:`low_latency_a2a_v2` - Low-latency version with FP8 quantization
- :doc:`/layers/nvidia/ep_a2a_layer` - High-level layer API
