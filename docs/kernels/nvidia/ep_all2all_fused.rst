Expert Parallelism All-to-All Fused Megakernel
================================================

Expert Parallelism (EP) All-to-All fused megakernel for single-node MoE models.
This kernel implements computation-communication fusion for dispatch+groupgemm and groupgemm+combine operations.

Overview
--------

The fused megakernel combines multiple operations into a single kernel launch to reduce kernel launch overhead,
improve SM utilization, and enable fine-grained task scheduling. It is specifically optimized for **single-node 8-GPU EP MoE** scenarios.

Key Features
^^^^^^^^^^^^

- **Megakernel Architecture**: Single kernel launch handles dispatch+groupgemm, and groupgemm+combine operations
- **Task-based Scheduling**: Dynamic task queue with atomic counter for load balancing
- **Token Saving/Skipping**: Optimizes communication by avoiding redundant token transfers
- **Token Sorting**: Reorders tokens to improve memory access patterns and enable early dispatch completion
- **SM Scheduling**: Fine-grained SM-level task distribution for optimal GPU utilization

Architecture
------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │              Mega Kernel Task Queue                        │
   │                                                             │
   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
   │  │ Dispatch     │  │ GroupGEMM    │  │ Combine       │   │
   │  │ Tasks        │  │ Tasks        │  │ Tasks         │   │
   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
   │         │                 │                 │            │
   │         └─────────────────┼─────────────────┘            │
   │                           │                               │
   │              ┌────────────▼────────────┐                 │
   │              │  Atomic Task Counter    │                 │
   │              │  (Shared across SMs)    │                 │
   │              └────────────┬────────────┘                 │
   │                           │                               │
   │         ┌─────────────────┼─────────────────┐           │
   │         │                 │                 │            │
   │    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐      │
   │    │  SM 0   │      │  SM 1   │ ...  │  SM N   │      │
   │    │         │      │         │      │         │      │
   │    │ Fetch   │      │ Fetch   │      │ Fetch   │      │
   │    │ Execute │      │ Execute │      │ Execute │      │
   │    └─────────┘      └─────────┘      └─────────┘      │
   └─────────────────────────────────────────────────────────────┘

Megakernel Functions
--------------------

Dispatch + GroupGEMM Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mega_kernel_dispatch_token_moe_grouped_gemm(...)

   Fused kernel that performs token dispatch and groupgemm computation in a single launch.

   **Key Components:**

   1. **Task Counter**: Atomic counter shared across all SMs for dynamic task distribution
   2. **Dispatch Tasks**: Token routing and communication operations
   3. **GroupGEMM Tasks**: Matrix multiplication for expert computation

   **Workflow:**

   .. code-block:: text

      while task_id < total_tasks:
          task_id = atomic_add(task_counter_ptr, 1)
          if task_id < num_dispatch_tasks:
              # Execute dispatch token operation
              tile_kernel_dispatch_token_intra_node(...)
          else:
              # Execute groupgemm operation
              tile_kernel_moe_grouped_gemm_nk_const(...)

   **Two-Stage Dispatch** (when ``NUM_TAIL_SMS > 0``):

   - **Stage 1**: Main dispatch SMs handle token routing
   - **Stage 2**: Tail SMs handle local copy and barrier notification
   - Enables overlap between dispatch completion and groupgemm start

GroupGEMM + Combine Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mega_kernel_moe_grouped_gemm_combine_token(...)

   Fused kernel that performs groupgemm computation and token combine in a single launch.

   **Two Modes:**

   1. **Serial Mode** (``USE_SCATTER_MODE=False``):
      
      - Execute all groupgemm tasks first
      - Synchronize with barrier_all
      - Execute combine tasks

   2. **Fuse Scatter Mode** (``USE_SCATTER_MODE=True``):
      
      - Interleave scatter, groupgemm, and reduce tasks
      - Fine-grained barrier synchronization per token block
      - Enables better overlap

   **Workflow (Fuse Scatter Mode):**

   .. code-block:: text

      while task_id < total_tasks:
          task_id = atomic_add(task_counter_ptr, 1)
          if task_id < num_combine_tasks:
              # Scatter tokens to output buffer
              tile_kernel_scatter_token_intra_node(...)
          elif task_id < num_combine_tasks + group_gemm_tasks:
              # Execute groupgemm with notification
              tile_kernel_moe_grouped_gemm_nk_const(..., NEED_NOTIFY=True)
          else:
              # TopK reduce
              tile_kernel_topk_reduce_token_intra_node(...)

Core Optimizations
------------------

Token Saving/Skipping
^^^^^^^^^^^^^^^^^^^^^^

**Problem**: When a token is routed to multiple experts on the same rank, we can avoid redundant communication
by only sending once and reusing the data.

**Solution**: Two-stage dispatch with token rank table and indirect position tracking.

**Implementation** (`tile_kernel_dispatch_token_intra_node_two_stage`):

.. code-block:: python

   # Stage 1: Main dispatch SMs
   for send_token_offset in range(global_warp_id, token_num * topk, total_warps):
       sort_token_offset = ld(token_sort_indices + send_token_offset)
       if sort_token_offset >= 0:  # ignore dropped tokens
           token_offset = sort_token_offset // topk
           expert_idx = ld(topk_indices_tensor + sort_token_offset)
           expert_rank = expert_idx // experts_per_rank
           
           # Check if token already sent to this rank
           has_sent = ld(token_rank_table_buf + token_offset * world_size + expert_rank)
           if has_sent < 0:
               # First time sending to this rank
               has_sent = store_idx
               libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)
               st(token_rank_table_buf + token_offset * world_size + expert_rank, store_idx)
           
           # Store indirect position for later lookup
           remote_token_indirect_pos = dl.symm_at(token_indirect_pos_buf, expert_rank)
           st(remote_token_indirect_pos + store_idx, has_sent, scope="sys", semantic="release")

   # Stage 2: Tail SMs copy locally and notify
   for tile_id in range(pid - num_pid, gemm_total_tiles_m, num_tail_sms):
       # Wait for indirect position
       has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
       while has_sent < 0:
           has_sent = ld_acquire(token_indirect_pos_buf + real_offset, scope="sys")
       
       # Copy from original position if different
       copy_warp(dispatch_output_local + real_offset * hidden_size, 
                 output_buf + has_sent * hidden_size, bytes_per_token)

**Benefits**:

- Reduces communication volume by ~30% for tokens with multiple same-rank experts
- Enables early groupgemm start while dispatch completes
- Maintains correctness through indirect position tracking

Token Sorting
^^^^^^^^^^^^^

**Problem**: Tokens arrive in arbitrary order, causing poor memory access patterns and delayed dispatch completion.

**Solution**: Pre-sort tokens by expert rank and expert index to improve locality.

**Implementation** (`get_ag_splits_and_recv_offset_for_dispatch`):

The preprocessing kernel generates ``token_sort_indices`` that reorders tokens:

.. code-block:: python

   # Sort tokens by: (expert_rank, expert_idx_intra_rank, token_idx)
   # This ensures:
   # 1. Tokens going to same expert are contiguous
   # 2. Memory access patterns are sequential
   # 3. Early completion signals can be sent per expert

   for token_idx in range(...):
       expert_idx = ld(topk_indices_buf + token_idx)
       expert_rank = expert_idx // experts_per_rank
       expert_idx_intra_rank = expert_idx % experts_per_rank
       
       # Compute sort key
       sort_key = (expert_rank * experts_per_rank + expert_idx_intra_rank) * max_tokens + token_idx
       st(token_sort_indices + sort_key, token_idx)

**Dispatch Loop** (`tile_kernel_dispatch_token_intra_node`):

.. code-block:: python

   # Iterate through sorted token offsets
   for send_token_offset in range(global_warp_id, token_num * topk, total_warps):
       sort_token_offset = ld(token_sort_indices + send_token_offset)
       if sort_token_offset >= 0:  # ignore dropped tokens
           token_offset = sort_token_offset // topk
           expert_idx = ld(topk_indices_tensor + sort_token_offset)
           # ... dispatch logic ...

**Benefits**:

- **Sequential Memory Access**: Tokens for same expert are processed together
- **Early Completion**: Can signal completion per expert as soon as all tokens arrive
- **Better Cache Utilization**: Improved L2 cache hit rate
- **Reduced Barrier Overhead**: Per-expert barriers instead of global barriers

SM Scheduling
^^^^^^^^^^^^^

**Problem**: Static task assignment leads to load imbalance and poor SM utilization.

**Solution**: Dynamic task queue with atomic counter for work-stealing.

**Task Distribution**:

.. code-block:: python

   # Each SM fetches tasks dynamically
   task_id = tl.atomic_add(task_counter_ptr, 1)
   while task_id < total_tasks:
       if task_id < num_dispatch_tasks:
           # Dispatch task
           execute_dispatch_task(task_id)
       elif task_id < num_dispatch_tasks + group_gemm_tasks:
           # GroupGEMM task
           execute_groupgemm_task(task_id - num_dispatch_tasks)
       else:
           # Combine task
           execute_combine_task(task_id - num_dispatch_tasks - group_gemm_tasks)
       
       # Fetch next task
       task_id = tl.atomic_add(task_counter_ptr, 1)

**SM Allocation Strategies**:

1. **Dispatch SMs** (``NUM_DISPATCH_SM``): Dedicated SMs for token dispatch
   - Typically 20-40 SMs depending on token count
   - Handles communication-intensive operations

2. **Tail SMs** (``NUM_TAIL_SMS``): SMs for two-stage dispatch completion
   - Typically 4-8 SMs
   - Handles local copy and barrier notification
   - Enables overlap with groupgemm

3. **GroupGEMM SMs**: Remaining SMs for computation
   - Automatically distributed via task queue
   - Load balanced across all experts

**Benefits**:

- **Load Balancing**: Fast SMs automatically take more tasks
- **Overlap**: Dispatch and groupgemm can overlap naturally
- **Flexibility**: Can adjust SM allocation based on workload characteristics

Barrier Synchronization
^^^^^^^^^^^^^^^^^^^^^^^

**Per-Expert Barriers** (Default):

.. code-block:: python

   # Signal completion when all tokens for an expert arrive
   sent_tokens = atomic_add(counter_ptr + expert_idx, 1)
   if sent_tokens == tokens_this_expert - 1:
       libshmem_device.fence()
       libshmem_device.signal_op(
           barriers_ptr + expert_idx_intra_rank * world_size + rank,
           1, libshmem_device.NVSHMEM_SIGNAL_SET, expert_rank)

**Block-wise Barriers** (``USE_BLOCK_WISE_BARRIER=True``):

.. code-block:: python

   # Per-tile barrier for finer granularity
   # Enables groupgemm to start as soon as a tile is ready
   barrier_idx = local_pid_m + tile_begin
   while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
       pass

**Per-Token Block Barriers** (Combine phase):

.. code-block:: python

   # Barrier per hidden_size / BLOCK_SIZE_N chunk
   barrier_n_idx = elem_idx * VEC_SIZE // BARRIER_TOKEN_BLOCK_SIZE
   barrier_idx = token_scatter_idx * N_BARRIERS_PER_TOKEN + barrier_n_idx
   token = ld_acquire(remote_barriers_ptr + barrier_idx, scope="sys")
   while token != 1:
       token = ld_acquire(remote_barriers_ptr + barrier_idx, scope="sys")

**Benefits**:

- **Finer Granularity**: Reduces wait time by enabling partial execution
- **Better Overlap**: GroupGEMM can start processing as soon as data is ready
- **Reduced Synchronization Overhead**: Smaller barrier scopes

Kernel Implementation Details
------------------------------

Dispatch Tile Kernel
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @triton_dist.jit(do_not_specialize=["pid", "num_pid"])
   def tile_kernel_dispatch_token_intra_node(
       pid, num_pid,
       counter_ptr, barriers_ptr,
       recv_buf_offset_per_expert,
       local_splits_buf,
       input_buf, output_buf,
       weight_send_buf, weight_recv_buf,
       topk_indices_tensor,
       token_dst_scatter_idx,
       num_input_tokens_per_rank,
       token_sort_indices,
       topk: tl.constexpr,
       hidden_size: tl.constexpr,
       experts_per_rank: tl.constexpr,
       HAS_WEIGHT: tl.constexpr,
       WITH_SCATTER_INDICES: tl.constexpr,
       num_warps: tl.constexpr,
       profiler: Profiler,
       ENABLE_PROFILING: tl.constexpr,
   ):
       WARP_SIZE = 32
       rank = dl.rank()
       world_size = dl.num_ranks()
       thread_idx = tid(0)
       lane_idx = thread_idx % WARP_SIZE
       warp_id = thread_idx // WARP_SIZE
       total_warps = num_warps * num_pid
       global_warp_id = pid * num_warps + warp_id
       
       token_num = tl.load(num_input_tokens_per_rank + rank)
       
       # Process tokens in sorted order
       for send_token_offset in range(global_warp_id, token_num * topk, total_warps):
           sort_token_offset = ld(token_sort_indices + send_token_offset)
           if sort_token_offset >= 0:  # ignore dropped tokens
               token_offset = sort_token_offset // topk
               expert_idx = ld(topk_indices_tensor + sort_token_offset)
               expert_rank = expert_idx // experts_per_rank
               expert_idx_intra_rank = expert_idx % experts_per_rank
               
               # Allocate destination slot
               if not WITH_SCATTER_INDICES:
                   store_idx = atomic_add_per_warp(
                       recv_buf_offset_per_expert + expert_rank * experts_per_rank * world_size +
                       expert_idx_intra_rank * world_size + rank, 1, 
                       scope="gpu", semantic="relaxed")
               else:
                   store_idx = ld(token_dst_scatter_idx + sort_token_offset)
               
               # Transfer token data
               src_ptr = input_buf + token_offset * hidden_size
               dst_ptr = output_buf + store_idx.to(tl.int64) * hidden_size
               libshmem_device.putmem_warp(dst_ptr, src_ptr, bytes_per_token, expert_rank)
               
               # Transfer weight if needed
               if HAS_WEIGHT:
                   libshmem_device.putmem_warp(
                       weight_recv_buf + store_idx, 
                       weight_send_buf + sort_token_offset,
                       weight_elem_size, expert_rank)
               
               # Signal completion per expert
               sync_warp()
               if lane_idx == 0:
                   tokens_this_expert = ld(local_splits_buf + expert_idx)
                   sent_tokens = atomic_add(counter_ptr + expert_idx, 1, 
                                          scope="gpu", semantic="relaxed")
                   if sent_tokens == tokens_this_expert - 1:
                       libshmem_device.fence()
                       libshmem_device.signal_op(
                           barriers_ptr + expert_idx_intra_rank * world_size + rank,
                           1, libshmem_device.NVSHMEM_SIGNAL_SET, expert_rank)

GroupGEMM Tile Kernel
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @triton_dist.jit(do_not_specialize=["pid", "num_pid", "M"])
   def tile_kernel_moe_grouped_gemm_nk_const(
       pid, num_pid,
       counter_ptr, barriers_ptr,
       a_ptr, b_ptr, c_ptr,
       expert_ids_ptr,
       split_size_ptr, split_size_cum_ptr,
       tile_num_ptr, tile_num_cum_ptr,
       num_total_tiles_ptr,
       M, N: tl.constexpr, K: tl.constexpr,
       stride_am, stride_ak,
       stride_be, stride_bn, stride_bk,
       stride_cm, stride_cn,
       BLOCK_SIZE_M: tl.constexpr,
       BLOCK_SIZE_N: tl.constexpr,
       BLOCK_SIZE_K: tl.constexpr,
       GROUP_SIZE_M: tl.constexpr,
       profiler: Profiler,
       NEED_WAIT: tl.constexpr,
       NEED_NOTIFY: tl.constexpr,
       USE_BLOCK_WISE_BARRIER: tl.constexpr,
       IS_DISPATCH_TWO_STAGET: tl.constexpr,
       ENABLE_PROFILING: tl.constexpr,
   ):
       num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
       pid_m = pid // num_block_n
       pid_n = pid % num_block_n
       
       expert_id = tl.load(expert_ids_ptr + pid_m)
       split_size = tl.load(split_size_ptr + expert_id)
       split_size_cum = tl.load(split_size_cum_ptr + pid_m)
       row_begin = split_size_cum
       
       # Wait for dispatch to complete (if needed)
       if NEED_WAIT:
           if IS_DISPATCH_TWO_STAGET:
               if USE_BLOCK_WISE_BARRIER:
                   barrier_idx = local_pid_m + tile_begin
                   while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                       pass
               else:
                   barrier_idx = expert_id
                   while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                       pass
           else:
               # Per-expert barrier
               barrier_idx = expert_id * world_size + thread_idx
               while ld_acquire(barriers_ptr + barrier_idx, scope="gpu") != 1:
                   pass
       
       # Execute GEMM
       # ... GEMM computation ...
       
       # Notify combine phase (if needed)
       if NEED_NOTIFY:
           __syncthreads()
           token_begin = row_begin + local_pid_m * BLOCK_SIZE_M
           valid_tokens = min(row_remain, BLOCK_SIZE_M)
           if thread_idx < valid_tokens:
               st(barriers_ptr + (token_begin + thread_idx) * num_block_n + pid_n, 
                  1, scope="gpu", semantic="release")

Combine Tile Kernels
^^^^^^^^^^^^^^^^^^^^^

**Gather Combine** (`tile_kernel_gather_combine_token_intra_node`):

- Reads expert outputs from remote ranks
- Accumulates outputs for each token's topk experts
- Uses vectorized operations (128-bit loads/stores)

**Scatter Combine** (`tile_kernel_scatter_token_intra_node`):

- Scatters expert outputs back to source ranks
- Waits for per-token-block barriers
- Transfers gate values if needed

**TopK Reduce** (`tile_kernel_topk_reduce_token_intra_node`):

- Reduces scattered outputs for each token
- Performs weighted sum across topk experts
- Final output per original token

Performance Characteristics
---------------------------

- **Num Tokens Per Rank**: 32k

**Single-Node 8-GPU Configuration**:

- **Communication**: NVSHMEM symmetric memory for direct GPU-to-GPU access
- **Latency**: ~3.5ms dispatch, ~5.9ms dispatch+groupgemm (depending on hidden size)
- **Throughput**: algorithm bandwith ~201GB/s (due to token saving, it exceeds hardware limit)

**Optimization Impact**:

- **Token Saving**:  ~18%
- **Token Sorting**: ~22%
- **Other Optimizations**: ~8%

Usage Example
-------------

.. code-block:: python

   from triton_dist.kernels.nvidia.ep_all2all_fused import (
       mega_kernel_dispatch_token_moe_grouped_gemm,
       mega_kernel_moe_grouped_gemm_combine_token,
   )
   
   # Dispatch + GroupGEMM
   mega_kernel_dispatch_token_moe_grouped_gemm[grid](
       task_counter_ptr,
       # ... dispatch params ...
       num_dispatch_tasks=NUM_DISPATCH_SM,
       # ... groupgemm params ...
       NUM_WARPS=16,
       NUM_TAIL_SMS=4,
       USE_BLOCK_WISE_BARRIER=True,
   )
   
   # GroupGEMM + Combine
   mega_kernel_moe_grouped_gemm_combine_token[grid](
       task_counter_ptr,
       # ... groupgemm params ...
       # ... combine params ...
       num_combine_tasks=COMBINE_SM,
       num_reduce_tasks=REDUCE_SM,
       USE_SCATTER_MODE=True,
       NUM_WARPS=32,
   )

See Also
--------

- :doc:`ep_a2a_intra_node` - Non-fused intra-node kernels
- :doc:`/layers/nvidia/ep_a2a_fused_layer` - High-level fused layer API
- :doc:`ep_a2a` - Inter-node EP All-to-All kernels

