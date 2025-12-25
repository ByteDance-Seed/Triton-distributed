Expert Parallelism All-to-All Fused Layer
===========================================

High-level layer API for fused Expert Parallelism All-to-All operations in single-node MoE models.
This layer combines dispatch, groupgemm, and combine operations into megakernels for optimal performance.

Overview
--------

``EpAll2AllFusedOp`` provides a complete fused solution for EP MoE operations, specifically optimized
for **single-node 8-GPU** configurations. It implements computation-communication fusion to minimize
kernel launch overhead and maximize GPU utilization.

Key Features
^^^^^^^^^^^^

- **Megakernel Fusion**: Single kernel launch for dispatch+groupgemm and groupgemm+combine
- **Token Optimization**: Token saving/skipping and sorting for reduced communication
- **Dynamic SM Scheduling**: Fine-grained task distribution across SMs
- **Two-Stage Dispatch**: Overlaps dispatch completion with groupgemm start
- **Fuse Scatter Mode**: Interleaves scatter, groupgemm, and reduce for better overlap
- **Lazy Memory Allocation**: Optional lazy NVSHMEM allocation for reduced startup time

Architecture
------------

.. code-block:: text

   EpAll2AllFusedOp
   ├── Preprocessing
   │   ├── get_ag_splits_and_recv_offset_for_dispatch()
   │   │   ├── Token sorting (token_sort_indices)
   │   │   ├── Expert split computation
   │   │   └── Scatter index generation
   │   └── Buffer initialization
   │
   ├── Mega Dispatch + GroupGEMM
   │   ├── mega_kernel_dispatch_token_moe_grouped_gemm()
   │   │   ├── Dispatch tasks (NUM_DISPATCH_SM SMs)
   │   │   │   ├── Token routing with token saving
   │   │   │   ├── Two-stage dispatch (if NUM_TAIL_SMS > 0)
   │   │   │   └── Per-expert barrier signaling
   │   │   └── GroupGEMM tasks (remaining SMs)
   │   │       ├── Wait for dispatch completion
   │   │       ├── Execute GEMM computation
   │   │       └── Notify combine phase
   │   └── Checkpoint (dispatch_output_local)
   │
   └── Mega GroupGEMM + Combine
       ├── mega_kernel_moe_grouped_gemm_combine_token()
       │   ├── GroupGEMM tasks
       │   │   ├── Execute GEMM computation
       │   │   └── Per-token-block barrier notification
       │   ├── Scatter tasks (fuse_scatter mode)
       │   │   ├── Scatter expert outputs
       │   │   └── Transfer gate values
       │   └── Reduce tasks
       │       └── TopK weighted sum
       └── Output copy

API Reference
-------------

EpAll2AllFusedOp
^^^^^^^^^^^^^^^^^

.. py:class:: EpAll2AllFusedOp(ep_group, max_tokens, hidden, topk, rank, num_tot_experts, local_world_size, world_size, dtype=torch.bfloat16, weight_dtype=torch.float32, num_sm=20, sm_margin=0, duplicate_comm_buffer=1, capacity=4.0, FWD_GEMM_BLOCK_SIZE_N=256, need_reversed_token_scatter_idx=False, lazy=False)

   Fused EP All-to-All layer for single-node MoE models.

   :param ep_group: PyTorch distributed process group for EP
   :param max_tokens: Maximum number of tokens per rank
   :param hidden: Hidden dimension size
   :param topk: Number of experts selected per token
   :param rank: Current rank
   :param num_tot_experts: Total number of experts across all ranks
   :param local_world_size: Number of ranks per node (must equal world_size for fused op)
   :param world_size: Total number of ranks (must equal local_world_size)
   :param dtype: Token data type (default: ``torch.bfloat16``)
   :param weight_dtype: Routing weight data type (default: ``torch.float32``)
   :param num_sm: Number of SMs to use for kernels (default: 20)
   :param sm_margin: SMs to reserve (default: 0)
   :param duplicate_comm_buffer: Number of communication buffers for pipelining (default: 1)
   :param capacity: Buffer capacity multiplier (default: 4.0)
   :param FWD_GEMM_BLOCK_SIZE_N: Block size N for forward GEMM (default: 256)
   :param need_reversed_token_scatter_idx: Whether to generate reverse scatter indices (default: False)
   :param lazy: Use lazy NVSHMEM allocation (default: False)

   .. note::

      This layer **only supports single-node** configurations (``world_size == local_world_size``).

   .. py:method:: preprocess(exp_indices, full_scatter_indices=None, local_scatter_indices=None)

      Preprocess expert indices and compute routing metadata.

      :param exp_indices: ``[num_tokens, topk]`` - Expert indices from Top-K gate
      :param full_scatter_indices: ``[num_tokens, topk]`` - Optional global scatter indices
      :param local_scatter_indices: ``[num_tokens, topk]`` - Optional local scatter indices
      :returns: ``EPAllToAllLayoutDesc`` - Layout descriptor for dispatch/combine

      **Key Operations**:

      1. **Bincount**: Count tokens per expert
      2. **AllGather Splits**: Exchange split information across ranks
      3. **Token Sorting**: Generate ``token_sort_indices`` for optimal memory access
      4. **Scatter Index Computation**: Compute destination offsets for each token

   .. py:method:: mega_dispatch_group_gemm(input, exp_indices, ep_a2a_layout_desc, gemm_weight, gemm_expert_ids, gemm_split_size, gemm_split_size_cum, gemm_tile_num, gemm_tile_num_cum, gemm_num_tiles_total, gemm_expert_offs, weight=None, with_cpy_flag=True, comm_buffer_id=0, optional_sm=None, num_tail_sms=0, gemm_input_reduce_last_dim=True, gemm_weight_reduce_last_dim=True, gemm_output_data=None, gemm_BLOCK_SIZE_N=256, gemm_BLOCK_SIZE_K=64, gemm_GROUP_SIZE_M=3, gemm_num_stages=3, use_block_wise_barrier=False, num_warps=16, enable_profiler=False, profile_file_name="mega_dispatch_group_gemm")

      Fused dispatch and groupgemm operation.

      :param input: ``[num_tokens, hidden]`` - Input tokens
      :param exp_indices: ``[num_tokens, topk]`` - Expert indices
      :param ep_a2a_layout_desc: ``EPAllToAllLayoutDesc`` - Layout descriptor from preprocess
      :param gemm_weight: ``[G, N, K]`` or ``[G, K, N]`` - Expert weights
      :param gemm_expert_ids: ``[M_grid]`` - Expert ID for each tile
      :param gemm_split_size: ``[G]`` - Token count per expert
      :param gemm_split_size_cum: ``[M_grid]`` - Cumulative token counts
      :param gemm_tile_num: ``[M_grid]`` - Number of tiles per expert
      :param gemm_tile_num_cum: ``[M_grid]`` - Cumulative tile counts
      :param gemm_num_tiles_total: ``[1]`` - Total number of tiles
      :param gemm_expert_offs: ``[experts_per_rank]`` - Expert offsets
      :param weight: ``[num_tokens, topk]`` - Optional routing weights
      :param with_cpy_flag: Whether to copy input to symmetric buffer (default: True)
      :param comm_buffer_id: Communication buffer ID for pipelining (default: 0)
      :param optional_sm: Override number of dispatch SMs (default: None)
      :param num_tail_sms: Number of tail SMs for two-stage dispatch (default: 0)
      :param gemm_input_reduce_last_dim: Whether input last dim is reduced (default: True)
      :param gemm_weight_reduce_last_dim: Whether weight last dim is reduced (default: True)
      :param gemm_output_data: Pre-allocated output buffer (default: None)
      :param gemm_BLOCK_SIZE_N: Block size N for GEMM (default: 256)
      :param gemm_BLOCK_SIZE_K: Block size K for GEMM (default: 64)
      :param gemm_GROUP_SIZE_M: Group size M for GEMM (default: 3)
      :param gemm_num_stages: Number of pipeline stages (default: 3)
      :param use_block_wise_barrier: Use per-tile barriers (default: False)
      :param num_warps: Number of warps per SM (default: 16)
      :param enable_profiler: Enable profiling (default: False)
      :param profile_file_name: Profile file name (default: "mega_dispatch_group_gemm")
      :returns: Tuple of (dispatch_output_local, weight_res, ep_a2a_layout_desc, gemm_output_data)

      **Workflow**:

      1. Copy input to symmetric buffer (if ``with_cpy_flag=True``)
      2. Initialize output buffer based on actual token counts
      3. Launch megakernel with dispatch and groupgemm tasks
      4. Return dispatch output (local copy) and groupgemm output

      **Two-Stage Dispatch** (when ``num_tail_sms > 0``):

      - **Main SMs**: Handle token routing and communication
      - **Tail SMs**: Handle local copy and barrier notification
      - Enables overlap between dispatch completion and groupgemm start

   .. py:method:: mega_group_gemm_combine(gemm_input_data, gemm_weight, gemm_expert_ids, gemm_split_size, gemm_split_size_cum, gemm_tile_num, gemm_tile_num_cum, gemm_num_tiles_total, ep_a2a_layout_desc, gemm_input_reduce_last_dim=True, gemm_weight_reduce_last_dim=True, gemm_BLOCK_SIZE_N=256, gemm_BLOCK_SIZE_K=64, gemm_GROUP_SIZE_M=3, gemm_num_stages=3, gate_input=None, cp_flag=True, combine_output=None, output_gate=None, optional_sm=None, num_reduce_sms=0, optional_signal_tensor=None, num_warps=32, combine_mode="serial", grad_output=None, orig_input=None, grad_weight=None, split_size_cum_per_expert=None, grad_BLOCK_SIZE_M=64, grad_BLOCK_SIZE_N=128, grad_BLOCK_SIZE_K=256, grad_GROUP_SIZE_M=3, enable_profiler=False, profile_file_name="mega_group_gemm_combine")

      Fused groupgemm and combine operation.

      :param gemm_input_data: ``[M, K]`` - Input data for groupgemm
      :param gemm_weight: ``[G, N, K]`` or ``[G, K, N]`` - Expert weights
      :param gemm_expert_ids: ``[M_grid]`` - Expert ID for each tile
      :param gemm_split_size: ``[G]`` - Token count per expert
      :param gemm_split_size_cum: ``[M_grid]`` - Cumulative token counts
      :param gemm_tile_num: ``[M_grid]`` - Number of tiles per expert
      :param gemm_tile_num_cum: ``[M_grid]`` - Cumulative tile counts
      :param gemm_num_tiles_total: ``[1]`` - Total number of tiles
      :param ep_a2a_layout_desc: ``EPAllToAllLayoutDesc`` - Layout descriptor from dispatch
      :param gemm_input_reduce_last_dim: Whether input last dim is reduced (default: True)
      :param gemm_weight_reduce_last_dim: Whether weight last dim is reduced (default: True)
      :param gemm_BLOCK_SIZE_N: Block size N for GEMM (default: 256)
      :param gemm_BLOCK_SIZE_K: Block size K for GEMM (default: 64)
      :param gemm_GROUP_SIZE_M: Group size M for GEMM (default: 3)
      :param gemm_num_stages: Number of pipeline stages (default: 3)
      :param gate_input: ``[recv_tokens]`` - Optional gate input values
      :param cp_flag: Whether to copy gate input (default: True)
      :param combine_output: Pre-allocated output buffer (default: None)
      :param output_gate: Pre-allocated gate output buffer (default: None)
      :param optional_sm: Override number of combine SMs (default: None)
      :param num_reduce_sms: Number of SMs for reduce phase (default: 0)
      :param optional_signal_tensor: Optional signal tensor (default: None)
      :param num_warps: Number of warps per SM (default: 32)
      :param combine_mode: Combine mode - "serial" or "fuse_scatter" (default: "serial")
      :param grad_output: ``[M, N]`` - Gradient output for backward (default: None)
      :param orig_input: ``[M, K]`` - Original input for backward (default: None)
      :param grad_weight: ``[G, N, K]`` - Gradient weight output (default: None)
      :param split_size_cum_per_expert: ``[G]`` - Cumulative split size per expert (default: None)
      :param grad_BLOCK_SIZE_M: Block size M for gradient GEMM (default: 64)
      :param grad_BLOCK_SIZE_N: Block size N for gradient GEMM (default: 128)
      :param grad_BLOCK_SIZE_K: Block size K for gradient GEMM (default: 256)
      :param grad_GROUP_SIZE_M: Group size M for gradient GEMM (default: 3)
      :param enable_profiler: Enable profiling (default: False)
      :param profile_file_name: Profile file name (default: "mega_group_gemm_combine")
      :returns: Combined output (and optionally gate output and grad_weight)

      **Combine Modes**:

      1. **Serial Mode** (``combine_mode="serial"``):
         
         - Execute all groupgemm tasks first
         - Synchronize with barrier_all
         - Execute combine tasks
         - Suitable for small token counts

      2. **Fuse Scatter Mode** (``combine_mode="fuse_scatter"``):
         
         - Interleave scatter, groupgemm, and reduce tasks
         - Fine-grained per-token-block barriers
         - Better overlap for large token counts
         - Recommended for production use

   .. py:method:: init_output_buffer(num_recv_tokens_per_rank, min_m=None)

      Initialize output buffer based on actual token counts.

      :param num_recv_tokens_per_rank: ``[world_size]`` - Token counts per rank (CPU pinned memory)
      :param min_m: Minimum M dimension for groupgemm (default: None)
      :returns: Tuple of (output_buf, weight_recv_buf)

      **Note**: This method polls CPU memory to avoid GPU-CPU synchronization overhead.

   .. py:method:: get_nvshmem_size()

      Get total NVSHMEM memory size in bytes.

   .. py:method:: get_nvshmem_size_gb()

      Get total NVSHMEM memory size in GB.

   .. py:method:: get_nvshmem_size_mb()

      Get total NVSHMEM memory size in MB.

   .. py:method:: get_nvshmem_breakdown()

      Get breakdown of NVSHMEM usage by buffer name.

   .. py:method:: sync()

      Materialize all NVSHMEM tensors (required if ``lazy=True``).

   .. py:method:: finalize()

      Release all NVSHMEM symmetric memory buffers.

EPAllToAllLayoutDesc
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: EPAllToAllLayoutDesc

   Layout descriptor containing routing metadata.

   .. py:attribute:: num_dispatch_token_cur_rank
      :type: int

      Number of tokens dispatched by this rank.

   .. py:attribute:: recv_buf_offset_per_expert
      :type: torch.Tensor

      ``[world_size, experts_per_rank, world_size]`` - Destination offsets for each token.

   .. py:attribute:: recv_buf_tokens_per_expert
      :type: torch.Tensor

      ``[world_size, experts_per_rank]`` - Token count per expert per rank.

   .. py:attribute:: num_recv_tokens_per_rank
      :type: torch.Tensor

      ``[world_size]`` - Total tokens received per rank.

   .. py:attribute:: num_input_tokens_per_rank
      :type: torch.Tensor

      ``[world_size]`` - Tokens dispatched per rank.

   .. py:attribute:: topk_indices_tensor
      :type: torch.Tensor

      ``[nnodes, max_tokens, topk]`` - Expert indices per token.

   .. py:attribute:: token_dst_scatter_idx
      :type: torch.Tensor

      ``[nnodes, max_tokens, topk]`` - Scatter indices in output buffer.

   .. py:attribute:: token_sort_indices
      :type: torch.Tensor

      ``[nnodes, max_tokens * topk]`` - Sorted token indices for optimal memory access.

   .. py:attribute:: reversed_token_scatter_idx
      :type: torch.Tensor

      ``[world_size * max_tokens * topk, 2]`` - Reverse mapping for combine phase.

Usage Example
-------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   import torch.distributed as dist
   from triton_dist.layers.nvidia.ep_a2a_fused_layer import EpAll2AllFusedOp
   
   # Initialize distributed runtime
   rank = dist.get_rank()
   world_size = dist.get_world_size()
   assert world_size == 8, "Fused op only supports single-node 8-GPU"
   
   # Create EP group
   ep_group = dist.group.WORLD
   
   # Create fused layer
   ep_op = EpAll2AllFusedOp(
       ep_group=ep_group,
       max_tokens=256,
       hidden=7168,
       topk=2,
       rank=rank,
       num_tot_experts=64,  # 64 experts / 8 ranks = 8 per rank
       local_world_size=8,
       world_size=8,
       dtype=torch.bfloat16,
       weight_dtype=torch.float32,
       num_sm=20,
       lazy=False,  # Set to True for lazy allocation
   )
   
   # Sync if using lazy allocation
   if ep_op._lazy:
       ep_op.sync()
   
   # Simulate input
   num_tokens = 128
   tokens = torch.randn(num_tokens, 7168, dtype=torch.bfloat16, device="cuda")
   expert_ids = torch.randint(0, 64, (num_tokens, 2), dtype=torch.int32, device="cuda")
   routing_weights = torch.softmax(torch.randn(num_tokens, 2, device="cuda"), dim=-1).float()
   
   # Preprocess
   layout_desc = ep_op.preprocess(
       exp_indices=expert_ids,
   )
   
   # Dispatch + GroupGEMM
   dispatch_output, weights, layout_desc, gemm_output = ep_op.mega_dispatch_group_gemm(
       input=tokens,
       exp_indices=expert_ids,
       ep_a2a_layout_desc=layout_desc,
       gemm_weight=expert_weights,  # [64, 7168, 2048]
       gemm_expert_ids=expert_ids_tiles,  # [M_grid]
       gemm_split_size=split_size,  # [64]
       gemm_split_size_cum=split_size_cum,  # [M_grid]
       gemm_tile_num=tile_num,  # [M_grid]
       gemm_tile_num_cum=tile_num_cum,  # [M_grid]
       gemm_num_tiles_total=num_tiles_total,  # [1]
       gemm_expert_offs=expert_offs,  # [8]
       weight=routing_weights,
       num_tail_sms=4,  # Enable two-stage dispatch
       use_block_wise_barrier=True,  # Use per-tile barriers
       num_warps=16,
   )
   
   # GroupGEMM + Combine
   combined_output = ep_op.mega_group_gemm_combine(
       gemm_input_data=gemm_output,  # Output from first groupgemm
       gemm_weight=expert_weights_2,  # [64, 2048, 7168]
       gemm_expert_ids=expert_ids_tiles,
       gemm_split_size=split_size,
       gemm_split_size_cum=split_size_cum,
       gemm_tile_num=tile_num,
       gemm_tile_num_cum=tile_num_cum,
       gemm_num_tiles_total=num_tiles_total,
       ep_a2a_layout_desc=layout_desc,
       gate_input=routing_weights,
       combine_mode="fuse_scatter",  # Use fuse scatter mode
       num_warps=32,
   )
   
   # Cleanup
   ep_op.finalize()

With Two-Stage Dispatch
^^^^^^^^^^^^^^^^^^^^^^^^

Two-stage dispatch enables overlap between dispatch completion and groupgemm start:

.. code-block:: python

   dispatch_output, weights, layout_desc, gemm_output = ep_op.mega_dispatch_group_gemm(
       ...,
       num_tail_sms=4,  # Allocate 4 SMs for tail operations
       use_block_wise_barrier=True,  # Required for two-stage dispatch
   )

**Benefits**:

- **Reduced Wait Time**: GroupGEMM can start as soon as a tile is ready
- **Better Overlap**: Dispatch completion overlaps with groupgemm execution
- **Improved Throughput**: 10-20% improvement for large token counts

With Fuse Scatter Mode
^^^^^^^^^^^^^^^^^^^^^^

Fuse scatter mode interleaves operations for better overlap:

.. code-block:: python

   combined_output = ep_op.mega_group_gemm_combine(
       ...,
       combine_mode="fuse_scatter",  # Enable fuse scatter mode
       num_reduce_sms=8,  # Allocate SMs for reduce phase
   )

**Benefits**:

- **Fine-Grained Barriers**: Per-token-block synchronization
- **Better Overlap**: Scatter, groupgemm, and reduce can overlap
- **Improved Latency**: 15-25% improvement for large token counts

With Lazy Allocation
^^^^^^^^^^^^^^^^^^^^

Lazy allocation defers NVSHMEM allocation until first use:

.. code-block:: python

   ep_op = EpAll2AllFusedOp(
       ...,
       lazy=True,  # Enable lazy allocation
   )
   
   # Query memory requirements before allocation
   print(f"NVSHMEM size: {ep_op.get_nvshmem_size_gb():.2f} GB")
   ep_op.print_nvshmem_breakdown()
   
   # Materialize when ready
   ep_op.sync()

**Benefits**:

- **Faster Startup**: Avoids allocation during initialization
- **Memory Planning**: Query requirements before allocation
- **Flexible**: Can adjust buffer sizes based on actual usage

Performance Tuning
------------------

SM Allocation
^^^^^^^^^^^^^

**Dispatch Phase**:

- **NUM_DISPATCH_SM**: 20-40 SMs (depends on token count)
- **NUM_TAIL_SMS**: 4-8 SMs (for two-stage dispatch)
- **Remaining SMs**: Automatically used for groupgemm

**Combine Phase**:

- **COMBINE_SM**: 20-40 SMs (depends on token count)
- **NUM_REDUCE_SMS**: 4-8 SMs (for fuse scatter mode)
- **Remaining SMs**: Automatically used for groupgemm

**Tuning Guidelines**:

- Start with ``num_sm=20`` and adjust based on profiling
- Use ``num_tail_sms=4`` for two-stage dispatch
- Use ``num_reduce_sms=8`` for fuse scatter mode
- Monitor SM utilization with profiling

GEMM Block Sizes
^^^^^^^^^^^^^^^^

**Forward GEMM**:

- **BLOCK_SIZE_N**: 256 (default, good for most cases)
- **BLOCK_SIZE_K**: 64 (default, good for most cases)
- **GROUP_SIZE_M**: 3 (default, balances parallelism and overhead)

**Gradient GEMM**:

- **grad_BLOCK_SIZE_M**: 64 (default)
- **grad_BLOCK_SIZE_N**: 128 (default)
- **grad_BLOCK_SIZE_K**: 256 (default)

**Tuning Guidelines**:

- Larger block sizes improve compute efficiency but reduce parallelism
- Smaller block sizes improve parallelism but increase overhead
- Use profiling to find optimal values for your workload

Warp Configuration
^^^^^^^^^^^^^^^^^^

- **Dispatch**: ``num_warps=16`` (good balance)
- **Combine**: ``num_warps=32`` (more parallelism for reduction)

**Tuning Guidelines**:

- More warps improve parallelism but reduce shared memory per warp
- Use ``num_warps=16`` for dispatch (communication-bound)
- Use ``num_warps=32`` for combine (computation-bound)

Profiling
---------

Enable profiling to analyze performance:

.. code-block:: python

   dispatch_output, weights, layout_desc, gemm_output = ep_op.mega_dispatch_group_gemm(
       ...,
       enable_profiler=True,
       profile_file_name="my_dispatch_profile",
   )
   
   combined_output = ep_op.mega_group_gemm_combine(
       ...,
       enable_profiler=True,
       profile_file_name="my_combine_profile",
   )

Profiles are saved to ``prof/mega/`` directory in Perfetto trace format.

**Key Metrics**:

- **dispatch_token_main**: Main dispatch time
- **dispatch_token_tail_notify**: Tail notification time
- **group_gemm_wait**: Wait time for dispatch completion
- **group_gemm_main**: GEMM computation time
- **combine_scatter_token**: Scatter time
- **combine_topk_reduce**: Reduce time

See Also
--------

- :doc:`/kernels/nvidia/ep_all2all_fused` - Underlying megakernel implementation
- :doc:`ep_a2a_layer` - Non-fused EP All-to-All layer
- :doc:`/kernels/nvidia/ep_a2a_intra_node` - Intra-node optimized kernels

