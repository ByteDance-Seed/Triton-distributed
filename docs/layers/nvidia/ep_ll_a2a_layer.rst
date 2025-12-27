Low-Latency EP All-to-All Layer
================================

High-level layer for ultra-low-latency Expert Parallelism All-to-All communication
with online FP8 quantization.

Overview
--------

``EPLowLatencyAllToAllLayer`` is optimized for latency-sensitive MoE inference scenarios,
achieving sub-202µs end-to-end All-to-All latency on 8 H800 GPUs.

Key Features
^^^^^^^^^^^^

- **Ultra-Low Latency**: ~202µs for 128 tokens per rank on 8 GPUs
- **Online FP8 Quantization**: 2x bandwidth reduction with per-group scaling
- **Double Buffering**: Overlaps computation and communication across iterations
- **Fused Operations**: Quantization, transfer, and postprocessing in single kernel
- **Built-in Profiling**: Detailed intra-kernel timing via Perfetto traces

Comparison with EPAll2AllLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - EPAll2AllLayer
     - EPLowLatencyAllToAllLayer
   * - Latency
     - Higher (multiple kernel launches)
     - Lower (~202µs for 8 GPUs)
   * - Throughput
     - Higher (larger batches)
     - Optimized for small batches
   * - Data Type
     - BF16/FP16
     - Online FP8 quantization
   * - Buffer Management
     - Dynamic resizing
     - Fixed-size double buffering
   * - Use Case
     - Training, large batch inference
     - Low-latency inference

Architecture
------------

Double Buffering
^^^^^^^^^^^^^^^^

.. code-block:: text

   Iteration 0        Iteration 1        Iteration 2
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Phase 0     │    │ Phase 1     │    │ Phase 0     │
   │ Buffer 0    │    │ Buffer 1    │    │ Buffer 0    │
   │ Signal=1    │    │ Signal=1    │    │ Signal=2    │
   └─────────────┘    └─────────────┘    └─────────────┘
   
   Signal cycling prevents false matches:
   - Phase 0: signal = call_count // 2 + 1
   - Phase 1: signal = call_count // 2 + 1

Memory Layout
^^^^^^^^^^^^^

.. code-block:: text

   Dispatch Buffers (per phase):
   ├── send_token_buffer    [max_m, msg_size]              # Local staging
   ├── recv_token_buffer    [experts/rank, world, max_m, msg_size]  # Symmetric
   ├── send_count_buffer    [world_size, experts/rank]     # Token counts
   ├── recv_count_buffer    [world_size, experts/rank]     # Symmetric
   ├── signal_buffer        [num_experts]                  # NVSHMEM signals
   └── recv_slot_counter    [num_experts]                  # Atomic counters
   
   Combine Buffers (per phase):
   ├── send_tokens_comm_buf [experts/rank, world * max_m, hidden]  # Symmetric
   ├── recv_token_buffer    [num_experts, max_m, hidden]   # Symmetric
   └── signal_buffer        [num_experts]                  # NVSHMEM signals

API Reference
-------------

EPLowLatencyAllToAllLayer
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: EPLowLatencyAllToAllLayer(max_m, hidden, topk, online_quant_fp8, rank, num_experts, local_world_size, world_size, fp8_gsize=128, dtype=torch.bfloat16, enable_profiling=False)

   Low-latency EP All-to-All layer with online FP8 quantization.

   :param max_m: Maximum tokens per rank (e.g., 128 or 256)
   :param hidden: Hidden dimension size
   :param topk: Number of experts per token
   :param online_quant_fp8: Must be True (only FP8 mode supported)
   :param rank: Current rank
   :param num_experts: Total number of experts
   :param local_world_size: Number of ranks per node
   :param world_size: Total number of ranks
   :param fp8_gsize: FP8 quantization group size (default: 128)
   :param dtype: Base data type for computation (default: ``torch.bfloat16``)
   :param enable_profiling: Enable intra-kernel profiling (default: False)

   .. py:method:: dispatch(send_tokens, send_scales, topk_indices)

      Dispatch tokens to experts with online FP8 quantization.

      :param send_tokens: ``[num_tokens, hidden]`` - Input tokens (bf16)
      :param send_scales: Must be None (online quantization only)
      :param topk_indices: ``[num_tokens, topk]`` - Expert routing decisions
      :returns: Tuple of (recv_token, recv_scale, expert_recv_count, dispatch_meta)

         - recv_token: ``[experts/rank, world * max_m, hidden]`` - FP8 tokens
         - recv_scale: ``[experts/rank, world * max_m, num_groups]`` - FP32 scales
         - expert_recv_count: ``[experts/rank]`` - Tokens per local expert
         - dispatch_meta: ``DispatchMetaInfo`` for combine

   .. py:method:: combine(send_tokens, topk_indices, topk_weights, dispatch_meta, zero_copy=False)

      Combine expert outputs with weighted reduction.

      :param send_tokens: ``[experts/rank, world * max_m, hidden]`` - Expert outputs
      :param topk_indices: ``[num_combined_tokens, topk]`` - Expert routing
      :param topk_weights: ``[num_combined_tokens, topk]`` - Routing weights
      :param dispatch_meta: Metadata from dispatch
      :param zero_copy: If True, assumes send_tokens is in comm buffer (default: False)
      :returns: ``[num_combined_tokens, hidden]`` - Combined output

   .. py:method:: finalize()

      Release NVSHMEM symmetric memory buffers.

   .. py:method:: dump_dispatch_trace()

      Export dispatch profiling to Perfetto trace.

      Output: ``./prof/ll_dispatch_RANK_{rank}.json``

   .. py:method:: dump_combine_trace()

      Export combine profiling to Perfetto trace.

      Output: ``./prof/ll_combine_RANK_{rank}.json``

DispatchMetaInfo
^^^^^^^^^^^^^^^^

.. py:class:: DispatchMetaInfo
   :no-index:

   Metadata from dispatch needed for combine operation.

   .. py:attribute:: recv_token_source_indices
      :type: torch.Tensor
      :no-index:

      ``[num_experts_per_rank, world_size * max_m]`` - Maps received positions to source token indices.

   .. py:attribute:: recv_token_source_count_and_start
      :type: torch.Tensor
      :no-index:

      ``[num_experts_per_rank, world_size]`` - Packed int64 containing (count, start) per source rank.

Usage Example
-------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   from triton_dist.utils import initialize_distributed
   from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer

   # Initialize distributed runtime
   rank, world_size = initialize_distributed()

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

   # Prepare inputs
   tokens = torch.randn(128, 7168, dtype=torch.bfloat16, device="cuda")
   expert_ids = torch.randint(0, 256, (128, 8), dtype=torch.int32, device="cuda")
   routing_weights = torch.softmax(torch.randn(128, 8, device="cuda"), dim=-1).float()

   # Dispatch (bf16 → fp8 quantized transfer)
   recv_token, recv_scale, expert_recv_count, dispatch_meta = layer.dispatch(
       send_tokens=tokens,
       send_scales=None,  # Online quantization
       topk_indices=expert_ids,
   )

   # Expert computation
   # recv_token: [experts/rank, world * max_m, hidden] - FP8
   # recv_scale: [experts/rank, world * max_m, num_groups] - FP32
   expert_output = your_fp8_expert_ffn(recv_token, recv_scale)

   # Combine
   combined = layer.combine(
       send_tokens=expert_output,
       topk_indices=expert_ids,
       topk_weights=routing_weights,
       dispatch_meta=dispatch_meta,
   )

   # Cleanup
   layer.finalize()

With Profiling
^^^^^^^^^^^^^^

.. code-block:: python

   # Enable profiling for performance analysis
   layer = EPLowLatencyAllToAllLayer(
       ...,
       enable_profiling=True,
   )

   # Run multiple iterations
   for _ in range(100):
       recv_token, recv_scale, expert_recv_count, dispatch_meta = layer.dispatch(...)
       combined = layer.combine(...)

   # Export traces (requires barrier for complete data)
   torch.distributed.barrier()
   layer.dump_dispatch_trace()
   layer.dump_combine_trace()

   # View in Perfetto UI (https://ui.perfetto.dev)

Expert FFN with FP8
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def fp8_expert_ffn(recv_token, recv_scale, expert_recv_count, expert_weights):
       """
       Process tokens with FP8 expert weights.
       
       recv_token: [experts/rank, world * max_m, hidden] - FP8
       recv_scale: [experts/rank, world * max_m, num_groups] - FP32
       expert_recv_count: [experts/rank] - Number of valid tokens per expert
       expert_weights: Expert FFN weights (may also be FP8)
       """
       outputs = []
       for expert_idx in range(num_experts_per_rank):
           count = expert_recv_count[expert_idx].item()
           if count == 0:
               continue
               
           # Extract valid tokens for this expert
           tokens = recv_token[expert_idx, :count]  # [count, hidden]
           scales = recv_scale[expert_idx, :count]  # [count, num_groups]
           
           # Dequantize and compute
           tokens_bf16 = dequantize_fp8(tokens, scales)
           output = expert_weights[expert_idx](tokens_bf16)
           
           # Store back
           outputs.append((expert_idx, output))
       
       # Reassemble output tensor
       return reassemble_outputs(outputs, recv_token.shape)

Performance Benchmark
---------------------

.. code-block:: bash

   # Run benchmark
   NVSHMEM_SYMMETRIC_SIZE=2g bash scripts/launch.sh \
       python/triton_dist/test/nvidia/test_ep_ll_a2a.py \
       -M 128 --iters 100 --verify-iters 20 --check

Expected results on 32x H800 (4 nodes):

.. code-block:: text

   Configuration:
   - Tokens per rank: 128
   - TopK: 8
   - Hidden size: 7168
   - Data type: FP8 (online quantization)
   
   Results:
   - Dispatch latency: ~70 µs (median)
   - Combine latency: ~67 µs (median)
   - Total A2A latency: ~137 µs

Profiling Categories
--------------------

**Dispatch Phases:**

- ``quant_and_put``: Online FP8 quantization and warp-level transfer
- ``count_put``: Exchange per-expert token counts
- ``wait``: Wait for all counts to arrive
- ``postprocess``: Reorganize received tokens by expert

**Combine Phases:**

- ``copy_and_put``: Copy to comm buffer and scatter to source ranks
- ``recv_wait``: Wait for all expert outputs
- ``topk_reduce``: Compute weighted sum of TopK expert outputs

.. note::

   This layer requires ``online_quant_fp8=True``. Pre-quantized input is not supported.
   For BF16/FP16 without quantization, use ``EPAll2AllLayer`` instead.

See Also
--------

- :doc:`/kernels/nvidia/low_latency_a2a_v2` - Underlying kernel implementation
- :doc:`ep_a2a_layer` - Standard EP layer (higher throughput, BF16 support)
