Expert Parallelism All-to-All (Intra-Node)
==========================================

Optimized intra-node All-to-All kernels for Expert Parallelism in MoE models.
These kernels leverage NVLink for high-bandwidth, low-latency communication within a single node.

Overview
--------

When all experts reside within a single node (``world_size == local_world_size``), 
intra-node optimized kernels can significantly reduce communication overhead by:

1. **Direct Symmetric Memory Access**: Use ``dl.symm_at()`` for direct GPU-to-GPU access
2. **NVLink Utilization**: Maximize NVLink bandwidth through warp-level operations
3. **Skipped Token Optimization**: Avoid redundant transfers when multiple TopK selections route to the same rank
4. **Local Combine Optimization**: Reduce tokens locally before final aggregation

Architecture
------------

.. code-block:: text

   Intra-Node Communication (8x GPUs with NVLink):
   
   ┌─────────────────────────────────────────────────────────┐
   │                    NVLink Mesh                          │
   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
   │  │GPU 0│═│GPU 1│═│GPU 2│═│GPU 3│═│GPU 4│═│GPU 5│═│GPU 6│═│GPU 7│ │
   │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
   │     │       │       │       │       │       │       │       │    │
   │     └───────┴───────┴───────┴───────┴───────┴───────┴───────┘    │
   │                    Symmetric Memory Pool                         │
   └─────────────────────────────────────────────────────────────────┘

   Key Optimizations:
   1. dl.symm_at(ptr, peer_rank) - Direct remote memory access
   2. putmem_signal_warp() - Warp-level transfer with signal
   3. Skipped token deduplication

Skipped Token Optimization
--------------------------

When a token is routed to multiple experts on the same rank, we only need to transfer 
it once and then duplicate locally:

.. code-block:: text

   Token with TopK=[Expert_0, Expert_2, Expert_5]
   
   If Expert_0 and Expert_2 are on Rank 1:
   - Only transfer token once to Rank 1
   - Store mapping: skipped_token_mapping_idx
   - During local dispatch: copy from first location
   
   This reduces NVLink bandwidth by (1 - 1/topk) for same-rank routing

Implementation:

.. code-block:: python

   # During dispatch: check if token was already sent to this rank
   for topk_idx in range(j):
       if expert_rank_of(topk_idx) == expert_rank:
           skip_this_token = True
           skipped_token_mapping_idx = token_dst_scatter_idx[topk_idx]
           break
   
   if not skip_this_token:
       # Full transfer with signal
       libshmem_device.putmem_signal_warp(dst_ptr, src_ptr, ...)
   else:
       # Just store the mapping index (no data transfer)
       st(dl.symm_at(mapping_indices + store_idx, expert_rank), 
          skipped_token_mapping_idx)

API Reference
-------------

Dispatch Kernels
^^^^^^^^^^^^^^^^

.. py:function:: kernel_dispatch_token_intra_node(dispatch_recv_token_num, intra_node_dispatch_skipped_token_mapping_indices, intra_node_dispatch_skipped_token_topk_mapping_indices, recv_buf_offset_per_expert, input_buf, output_buf, weight_send_buf, weight_recv_buf, topk_indices_tensor, token_dst_scatter_idx, num_input_tokens_per_rank, topk, hidden_size, bytes_per_token, experts_per_rank, local_world_size, HAS_WEIGHT, WITH_SCATTER_INDICES, num_warps)

   Intra-node token dispatch with skipped token optimization.

   :param dispatch_recv_token_num: Number of tokens to receive (for output buffer bounds)
   :param intra_node_dispatch_skipped_token_mapping_indices: ``[local_world_size * max_tokens * topk]`` - 
       Stores the first scatter index when token is sent to same rank multiple times (symmetric)
   :param intra_node_dispatch_skipped_token_topk_mapping_indices: ``[local_world_size * max_tokens * topk, topk]`` - 
       Per-TopK mapping for skipped tokens (symmetric)
   :param recv_buf_offset_per_expert: ``[world_size, experts_per_rank, world_size]`` - Destination offsets
   :param input_buf: Source token buffer
   :param output_buf: Destination buffer (symmetric)
   :param weight_send_buf: Optional routing weights source
   :param weight_recv_buf: Optional routing weights destination
   :param topk_indices_tensor: ``[max_tokens, topk]`` - Expert indices per token
   :param token_dst_scatter_idx: ``[max_tokens, topk]`` - Output scatter indices
   :param num_input_tokens_per_rank: ``[world_size]`` - Token count per rank
   :param experts_per_rank: Number of experts per rank (constexpr)
   :param local_world_size: Number of ranks in node (constexpr)
   :param HAS_WEIGHT: Whether to transfer weights (constexpr)
   :param WITH_SCATTER_INDICES: Whether scatter indices are precomputed (constexpr)

.. py:function:: kernel_skipped_token_local_dispatch_intra_node(dispatch_recv_token_num, intra_node_dispatch_skipped_token_mapping_indices, intra_node_dispatch_skipped_token_topk_mapping_indices, intra_node_dispatch_skipped_token_mapping_indices_copy, intra_node_dispatch_skipped_token_topk_mapping_indices_copy, dispatch_out_buf, hidden_size, bytes_per_token, topk, ENABLE_LOCAL_COMBINE, num_warps)

   Post-dispatch local copy for skipped tokens.

   After the main dispatch, tokens that were "skipped" (because another TopK selection 
   already sent the same token to this rank) need to be locally copied from the first 
   location to their expected positions.

   :param dispatch_recv_token_num: Number of received tokens
   :param dispatch_out_buf: Token buffer with received data
   :param ENABLE_LOCAL_COMBINE: Whether to prepare for local combine optimization

Combine Kernels
^^^^^^^^^^^^^^^

.. py:function:: kernel_combine_token_intra_node(num_input_tokens_per_rank, input_buf, send_buf, topk_indices_buf, token_dst_scatter_idx, max_tokens, topk, hidden_size, bytes_per_token, expert_per_rank, local_world_size, ENABLE_LOCAL_COMBINE, num_warps)

   Combine expert outputs within a single node.

   Uses direct symmetric memory access to read from peer GPUs and accumulate 
   outputs for each original token.

   :param num_input_tokens_per_rank: ``[world_size]`` - Token count per rank
   :param input_buf: Expert output buffer (symmetric) - read from peers
   :param send_buf: ``[nnodes, max_tokens, hidden]`` - Combined output
   :param topk_indices_buf: ``[max_tokens, topk]`` - Expert indices per token
   :param token_dst_scatter_idx: ``[max_tokens, topk]`` - Scatter indices from dispatch
   :param ENABLE_LOCAL_COMBINE: Skip tokens already combined locally

   Vectorized accumulation:

   .. code-block:: python

      for i in range(lane_id * vec_size, hidden_size, WARP_SIZE * vec_size):
          token_accum = dl.zeros_vector(vec_size, tl.float32)
          for j in range(topk):
              if expert_node_idx == node_id:
                  remote_input_ptr = dl.symm_at(input_buf, expert_rank)
                  token = dl.ld_vector(remote_input_ptr + offset, vec_size=vec_size)
                  token_accum = token_accum + token.to(tl.float32)
          dl.st_vector(send_buf + out_offset, token_accum.to(send_buf.dtype))

.. py:function:: kernel_skipped_token_inplace_local_combine_intra_node(combine_token_num, intra_node_dispatch_skipped_token_mapping_indices, skipped_token_topk_mapping_indices, combine_input_buf, hidden_size, topk, num_warps)

   Pre-combine local reduction for tokens with multiple same-rank experts.

   When ENABLE_LOCAL_COMBINE is True, tokens that were routed to multiple experts 
   on the same rank are pre-aggregated in-place before the main combine phase.

   :param combine_token_num: Number of tokens to process
   :param intra_node_dispatch_skipped_token_mapping_indices: Mapping to first token location
   :param skipped_token_topk_mapping_indices: Per-TopK mappings
   :param combine_input_buf: Expert output buffer (modified in-place)

Preprocessing Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: get_ag_splits_and_recv_offset_for_dispatch_intra_node(topk_indices, local_splits, full_splits_buf, splits_signal_buf, topk, local_world_size, world_size, max_tokens, experts_per_rank, full_scatter_indices=None, cpu_default_val=-1, offset_dtype=torch.int32, num_sm=20)

   Compute routing metadata for intra-node dispatch.

   This is a simplified version that doesn't need inter-node communication.

   :returns: Tuple of (recv_buf_offset_per_expert, num_recv_tokens_per_rank_cpu, 
             num_input_tokens_per_rank, token_dst_scatter_idx)

Implementation Details
----------------------

Warp-Level Transfer with Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kernel uses ``putmem_signal_warp`` to atomically transfer data and set a signal:

.. code-block:: python

   # Transfer token data and set signal in one operation
   libshmem_device.putmem_signal_warp(
       dst_ptr,                    # Remote destination
       src_ptr,                    # Local source
       bytes_per_token,            # Transfer size
       mapping_indices + store_idx, # Signal address
       skipped_token_mapping_idx,  # Signal value (maps to first occurrence)
       libshmem_device.NVSHMEM_SIGNAL_SET,
       expert_rank                 # Target rank
   )

Grid Synchronization
^^^^^^^^^^^^^^^^^^^^

Uses GPU-wide barriers for coordination:

.. code-block:: python

   barrier_on_this_grid(grid_sync_counter, False)
   if pid == 0:
       libshmem_device.barrier_all_block()
   barrier_on_this_grid(grid_sync_counter, False)

Performance Characteristics
---------------------------

- **NVLink Bandwidth**: Achieves near-peak NVLink bandwidth (~900 GB/s on H100)
- **Low Latency**: Direct memory access eliminates host involvement
- **Skipped Token Savings**: Reduces bandwidth by up to ``(topk-1)/topk`` for same-rank routing
- **Local Combine**: Pre-aggregation reduces combine phase work

.. note::

   Intra-node kernels are automatically selected when ``world_size == local_world_size``.
   For multi-node scenarios, these kernels handle the intra-node portion while 
   ``ep_a2a.py`` handles inter-node communication.

Usage Example
-------------

The intra-node kernels are used automatically by ``EPAll2AllLayer`` when appropriate:

.. code-block:: python

   # Single-node setup (world_size == local_world_size)
   ep_layer = EPAll2AllLayer(
       ep_group=ep_group,
       max_tokens=256,
       hidden=7168,
       topk=8,
       rank=rank,
       num_tot_experts=64,        # 8 experts per GPU
       local_world_size=8,
       world_size=8,              # Single node
       enable_local_combine=True, # Enable local combine optimization
   )

   # Dispatch and combine use intra-node optimized kernels
   output, weights, layout_desc = ep_layer.dispatch(tokens, expert_ids, routing_weights)
   combined = ep_layer.combine(expert_output, layout_desc)

Run Example
-----------

.. code-block:: sh
   
   NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 32768 -N 1536 --topk 8 -G 384 --drop_ratio 0.3 --enable-local-combine --check


See Also
--------

- :doc:`ep_a2a` - Main EP A2A kernels (handles inter-node)
- :doc:`low_latency_a2a_v2` - Low-latency version with FP8 quantization
- :doc:`/layers/nvidia/ep_a2a_layer` - High-level layer API

