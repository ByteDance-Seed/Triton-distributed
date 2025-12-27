Low-Latency All-to-All V2 (EP)
==============================

Ultra-low-latency All-to-All communication kernels for Expert Parallelism with online FP8 quantization.
Designed for latency-sensitive MoE inference scenarios.

Overview
--------

The Low-Latency All-to-All V2 kernels are optimized for:

1. **Minimal Latency**: Single-kernel dispatch and combine operations
2. **Online FP8 Quantization**: Reduces transfer size by 2x with minimal accuracy loss
3. **Double Buffering**: Overlaps communication with computation across iterations
4. **Fused Operations**: Quantization, transfer, and postprocessing in single kernel launch

Performance
^^^^^^^^^^^

.. code-block:: text

   Benchmark: 8x H800 GPUs
   - Tokens per rank: 128
   - TopK: 8
   - Hidden size: 7168
   - Data type: FP8 (online quantization)
   
   Results:
   - Dispatch latency: ~76 µs
   - Combine latency: ~126 µs
   - Total A2A latency: ~202 µs

Architecture
------------

Message Format
^^^^^^^^^^^^^^

Each token message contains:

.. code-block:: text

   ┌─────────────┬──────────────────────────────────────┬───────────────┐
   │  META (16B) │           TOKEN (hidden × 1B)        │ SCALE (groups × 4B) │
   │  (padded)   │           (FP8 quantized)            │    (FP32)    │
   └─────────────┴──────────────────────────────────────┴───────────────┘
   
   META: Source token index (int32, padded to 16 bytes for alignment)
   TOKEN: FP8-quantized hidden state
   SCALE: Per-group quantization scales (hidden // fp8_gsize groups)

Double Buffering
^^^^^^^^^^^^^^^^

.. code-block:: text

   Phase 0:                    Phase 1:
   ┌─────────────────┐        ┌─────────────────┐
   │  Buffer Set 0   │        │  Buffer Set 1   │
   │  - send_token   │        │  - send_token   │
   │  - recv_token   │        │  - recv_token   │
   │  - signal_buf   │        │  - signal_buf   │
   └─────────────────┘        └─────────────────┘
          ↑                          ↑
          │                          │
   Iteration 0,2,4...        Iteration 1,3,5...
   
   Benefits:
   - No explicit synchronization between iterations
   - Signal values cycle: 1,1,2,2,3,3,...
   - Enables compute-communication overlap

Dispatch Kernel (V2)
--------------------

The dispatch kernel performs:

1. **Online FP8 Quantization**: Per-group scaling and quantization
2. **Warp-level Transfer**: Parallel token transfers to target experts
3. **Count Exchange**: AllGather of per-expert token counts
4. **Postprocessing**: Reorganize received tokens by expert

.. code-block:: python

   @triton_dist.jit
   def dispatch_kernel_v2(
       profiler_buf,
       send_tensor,              # [num_tokens, HIDDEN] - input tokens (bf16)
       send_scale,               # [num_tokens, NUM_GROUPS] - optional precomputed scales
       topk_idx,                 # [num_tokens, TOPK] - expert routing
       num_tokens,
       send_token_buffer,        # [max_m, msg_size] - symmetric
       recv_token_buffer,        # [num_experts_per_rank, world_size, max_m, msg_size] - symmetric
       send_count_buffer,        # [world_size, num_experts_per_rank] - symmetric
       recv_count_buffer,        # [world_size, num_experts_per_rank] - symmetric
       recv_slot_counter,        # [num_experts] - atomic counters
       signal_buffer,            # [WORLD_SIZE] - NVSHMEM signals
       recv_token_source_indices,    # [num_experts_per_rank, world_size * max_m] - output
       recv_scale,               # [num_experts_per_rank, world_size * max_m, num_groups] - output
       recv_token,               # [num_experts_per_rank, world_size * max_m, hidden] - output
       expert_recv_count,        # [num_experts_per_rank] - output
       recv_token_source_count_and_start,  # [num_experts_per_rank, world_size] - output
       grid_sync_counter,
       signal_val: int,
       ...
   ):

Kernel Phases
^^^^^^^^^^^^^

**Phase 0: Quantize and Transfer**

.. code-block:: python

   # Per-token processing
   for i in range(pid, num_tokens, num_ctas):
       cur_token = tl.load(send_token_ptrs, pertoken_mask)
       
       # Online FP8 quantization (per-group)
       group = tl.reshape(cur_token, (BLOCK_SCALE, FP8_GSIZE))
       scale = tl.max(tl.abs(group), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
       quant = (group.to(tl.float32) / scale).to(tl.float8e4nv)
       
       # Store token index as metadata
       tl.store(tl.cast(send_buffer, tl.pointer_type(tl.int32)), i)
       tl.store(send_token_buffer_ptrs, quant)
       tl.store(send_scale_buffer_ptrs, scale)
       
       # Warp-level transfer to each TopK expert
       for warp_id in range(TOPK):
           dst_expert = tl.load(topk_idx_ptrs + warp_id)
           dst_slot = atomic_add_per_warp(recv_slot_counter + dst_expert, 1)
           libshmem_device.putmem_nbi_warp(recv_buffer + dst_slot * MSG_SIZE, 
                                            send_buffer, MSG_SIZE, dst_rank)

**Phase 1: Exchange Counts**

.. code-block:: python

   barrier_on_this_grid(grid_sync_counter, False)
   libshmem_device.fence()
   
   for dst_rank in range(pid, WORLD_SIZE, num_ctas):
       token_counts = tl.load(recv_slot_counter + dst_rank * NUM_EXPERTS_PER_RANK + ...)
       libshmem_device.putmem_signal_nbi_block(
           recv_count_buffer + rank * NUM_EXPERTS_PER_RANK,
           send_count_buffer + dst_rank * NUM_EXPERTS_PER_RANK,
           NUM_EXPERTS_PER_RANK * 4,
           signal_buffer + rank, signal_val,
           libshmem_device.NVSHMEM_SIGNAL_SET, dst_rank)

**Phase 2: Wait for All Counts**

.. code-block:: python

   for src_rank in range(pid, WORLD_SIZE, num_ctas):
       libshmem_device.signal_wait_until(
           signal_buffer + src_rank,
           libshmem_device.NVSHMEM_CMP_EQ,
           signal_val)

**Phase 3: Postprocess**

.. code-block:: python

   # Reorganize tokens by expert
   for target_expert_idx in range(pid, NUM_EXPERTS, num_ctas):
       dispatch_postprocess_kernel_v2_for_expert(
           target_expert_idx,
           recv_token_source_indices, recv_scale, recv_token,
           expert_recv_count, recv_token_source_count_and_start,
           recv_token_buffer, recv_count_buffer, ...)

Combine Kernel (V2)
-------------------

The combine kernel performs:

1. **Copy to Communication Buffer**: Prepare tokens for transfer
2. **Scatter to Source Ranks**: Send expert outputs back to original token positions
3. **Wait for All Data**: Synchronize all transfers
4. **TopK Weighted Reduce**: Compute final combined output

.. code-block:: python

   @triton_dist.jit
   def combine_kernel_v2(
       profiler_buf,
       send_tokens,              # [num_experts_per_rank, world_size * max_m, hidden]
       send_tokens_comm_buf,     # Communication buffer (symmetric)
       topk_indices,             # [num_combined_tokens, topk]
       topk_weights,             # [num_combined_tokens, topk]
       combined_out,             # [num_combined_tokens, hidden] - output
       recv_token_buffer,        # [num_experts, max_m, hidden] - symmetric
       signal_buf,               # [num_expert] - NVSHMEM signals
       dispatch_recv_token_source_indices,
       dispatch_recv_token_source_count_and_start,
       grid_sync_counter,
       num_combined_tokens: int,
       signal_val: int,
       ...
   ):

Intra-Node Optimization
^^^^^^^^^^^^^^^^^^^^^^^

For intra-node transfers, the kernel uses direct symmetric memory access instead of NVSHMEM put:

.. code-block:: python

   is_intra_node = (dst_rank // LOCAL_WORLD_SIZE) == cur_node_id

   if is_intra_node:
       # Direct load/store through symmetric memory
       dst_remote_ptr = dl.symm_at(dst_ptr, dst_rank)
       for h_idx in range(lane_id, num_hidden_iters, WARP_SIZE):
           val_vec = dl.ld_vector(src_ptr + h_idx * VEC_SIZE, vec_size=VEC_SIZE)
           dl.st_vector(dst_remote_ptr + h_idx * VEC_SIZE, val_vec)
   else:
       # NVSHMEM put for inter-node
       libshmem_device.putmem_nbi_warp(dst_ptr, src_ptr, nbytes, dst_rank)

API Reference
-------------

Context Management
^^^^^^^^^^^^^^^^^^

.. py:function:: create_ep_ll_a2a_ctx(max_m, hidden, topk, num_experts, online_quant_fp8, fp8_gsize, dtype, world_size, rank)

   Create dispatch and combine contexts for low-latency EP All-to-All.

   :param max_m: Maximum tokens per rank (e.g., 128 or 256)
   :param hidden: Hidden dimension size
   :param topk: Number of experts per token
   :param num_experts: Total number of experts
   :param online_quant_fp8: Must be True (only FP8 mode supported)
   :param fp8_gsize: FP8 quantization group size (default: 128)
   :param dtype: Base data type (e.g., torch.bfloat16)
   :param world_size: Total number of ranks
   :param rank: Current rank
   :returns: Tuple of (LowlatencyDispatchContext, LowlatencyCombineContext)

Data Structures
^^^^^^^^^^^^^^^

.. py:class:: LowlatencyDispatchContext

   Context for low-latency dispatch operations.

   .. py:attribute:: signal_val
      :type: int

      Current signal value for this phase (cycles: 1,1,2,2,3,3,...)

   .. py:method:: update_phase()

      Advance to next phase (toggles buffer set and updates signal value).

   .. py:method:: finalize()

      Release NVSHMEM symmetric memory buffers.

.. py:class:: LowlatencyCombineContext

   Context for low-latency combine operations.

   Same interface as LowlatencyDispatchContext.

.. py:class:: DispatchMetaInfo

   Metadata from dispatch operation needed for combine.

   .. py:attribute:: recv_token_source_indices
      :type: torch.Tensor

      ``[num_experts_per_rank, world_size * max_m]`` - Maps received tokens to source indices.

   .. py:attribute:: recv_token_source_count_and_start
      :type: torch.Tensor

      ``[num_experts_per_rank, world_size]`` - Packed (count, start) per source rank.

Kernel Functions
^^^^^^^^^^^^^^^^

.. py:function:: dispatch_kernel_v2[grid](profiler_buf, send_tensor, send_scale, topk_idx, num_tokens, ...)

   Low-latency dispatch kernel with online FP8 quantization.

   :param send_tensor: ``[num_tokens, HIDDEN]`` - Input tokens (bf16)
   :param topk_idx: ``[num_tokens, TOPK]`` - Expert routing decisions
   :param ONLINE_QUANT_FP8: Whether to perform online FP8 quantization (must be True)
   :param FP8_GSIZE: Quantization group size (typically 128)
   :param ENABLE_PROFILING: Enable intra-kernel profiling

.. py:function:: combine_kernel_v2[grid](profiler_buf, send_tokens, send_tokens_comm_buf, topk_indices, topk_weights, combined_out, ...)

   Low-latency combine kernel with weighted reduction.

   :param send_tokens: ``[num_experts_per_rank, world_size * max_m, hidden]`` - Expert outputs
   :param topk_weights: ``[num_combined_tokens, topk]`` - Routing weights
   :param combined_out: ``[num_combined_tokens, hidden]`` - Output buffer
   :param ZERO_COPY: If True, assumes send_tokens is already in comm buffer

Profiling Support
-----------------

Built-in profiling for performance analysis:

.. code-block:: python

   # Enable profiling
   layer = EPLowLatencyAllToAllLayer(..., enable_profiling=True)

   # Run operations
   recv_token, recv_scale, expert_recv_count, dispatch_meta = layer.dispatch(...)
   combined = layer.combine(...)

   # Export traces
   layer.dump_dispatch_trace()  # Outputs: ./prof/ll_dispatch_RANK_X.json
   layer.dump_combine_trace()   # Outputs: ./prof/ll_combine_RANK_X.json

Profiling categories:

- **Dispatch**: ``quant_and_put``, ``count_put``, ``wait``, ``postprocess``
- **Combine**: ``copy_and_put``, ``recv_wait``, ``topk_reduce``

Usage Example
-------------

.. code-block:: python

   from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer

   # Create layer
   layer = EPLowLatencyAllToAllLayer(
       max_m=128,
       hidden=7168,
       topk=8,
       online_quant_fp8=True,
       rank=rank,
       num_experts=256,
       local_world_size=8,
       world_size=32,
       fp8_gsize=128,
       dtype=torch.bfloat16,
       enable_profiling=False,
   )

   # Dispatch (bf16 input → fp8 quantized transfer)
   recv_token, recv_scale, expert_recv_count, dispatch_meta = layer.dispatch(
       send_tokens=tokens,        # [num_tokens, hidden], bf16
       send_scales=None,          # Online quantization
       topk_indices=expert_ids,   # [num_tokens, topk]
   )

   # Expert computation on recv_token (fp8 with scales)
   # ... your expert FFN here ...
   expert_output = ...  # [num_experts_per_rank, world_size * max_m, hidden]

   # Combine
   combined = layer.combine(
       send_tokens=expert_output,
       topk_indices=expert_ids,
       topk_weights=routing_weights,
       dispatch_meta=dispatch_meta,
   )

   # Cleanup
   layer.finalize()

.. note::

   This kernel currently requires ``online_quant_fp8=True``. 
   Pre-quantized input mode is not yet supported.

Run Example
-----------

.. code-block:: sh
   
   NVSHMEM_SYMMETRIC_SIZE=2g bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_ll_a2a.py -M 128 --profile



See Also
--------

- :doc:`ep_a2a` - Standard EP A2A kernels (higher throughput, higher latency)
- :doc:`ep_a2a_intra_node` - Intra-node optimized kernels
- :doc:`/layers/nvidia/ep_ll_a2a_layer` - High-level layer API

