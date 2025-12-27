Triton-distributed Semantics
============================

This document describes the design philosophy and semantic model of Triton-distributed.

Design Philosophy
-----------------

Triton-distributed is built on a **Tile-Centric** design philosophy (as described in our `MLSys 2025 paper <https://arxiv.org/abs/2503.20313>`_), which means:

1. **Tile as the Unit of Work**: Both computation and communication are organized around tiles (blocks of data). Each tile is a self-contained unit that can be computed, transferred, and synchronized independently.

2. **Decoupled Computation and Communication**: Communication (data transfer) and computation (GEMM, etc.) are explicitly separated and can be performed by different SMs (Streaming Multiprocessors) or different parts of the same kernel.

3. **Fine-grained Overlapping**: By organizing work around tiles, we can achieve fine-grained overlapping where computation on tiles that are already available can proceed while other tiles are still being transferred.

Core Semantic Concepts
----------------------

Producer-Consumer Model
~~~~~~~~~~~~~~~~~~~~~~~

Triton-distributed uses a **producer-consumer** model for overlapping computation with communication:

- **Producer**: Responsible for data transfer (e.g., AllGather, All-to-All). The producer copies data to a shared buffer and signals when each tile is ready.

- **Consumer**: Responsible for computation (e.g., GEMM). The consumer waits for tiles to be ready, then computes on them immediately.

.. code-block:: python

    # Producer: Transfer data and signal
    @triton_dist.jit
    def producer_kernel(...):
        # Transfer tile to shared buffer
        tl.store(remote_buffer_ptr + tile_offset, data)
        # Signal that tile is ready
        dl.notify(signal_ptr, peer_rank, signal=tile_id, sig_op="set")

    # Consumer: Wait for data and compute
    @triton_dist.jit
    def consumer_kernel(...):
        # Wait for tile to be ready
        token = dl.wait(signal_ptr + tile_id, 1, "gpu", "acquire")
        # Consume token to establish data dependency
        data_ptr = dl.consume_token(data_ptr, token)
        # Now safe to compute on the tile
        result = compute(tl.load(data_ptr))

Signal-Based Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synchronization model is based on **signals** rather than barriers:

- ``wait(ptr, n, scope, semantic)``: Wait until n signals at ptr reach expected values
- ``notify(ptr, rank, signal, sig_op)``: Send a signal to a specific rank
- ``consume_token(value, token)``: Establish data dependency between wait and memory access

**Key Insight**: Unlike global barriers that synchronize all ranks, signal-based synchronization allows fine-grained tile-level coordination. This enables:

- Different tiles to proceed independently
- Maximum overlapping between communication and computation
- Avoiding the "straggler effect" where slow ranks block everyone

Memory Scope and Ordering
~~~~~~~~~~~~~~~~~~~~~~~~~

Triton-distributed provides explicit control over memory scope and ordering:

**Scope**:

- ``"gpu"``: Synchronization visible only within the same GPU
- ``"sys"``: Synchronization visible across the entire system (multiple GPUs/nodes)

**Semantic**:

- ``"acquire"``: Ensures all previous writes by other threads are visible after the wait
- ``"release"``: Ensures all previous writes by this thread are visible to other threads

Symmetric Memory Model
~~~~~~~~~~~~~~~~~~~~~~

Triton-distributed uses **symmetric memory** (NVSHMEM/ROCSHMEM) for cross-rank communication:

- All ranks allocate memory at the same virtual address
- ``symm_at(ptr, rank)``: Maps a local pointer to the corresponding address on another rank
- Enables direct load/store across ranks without explicit send/receive

.. code-block:: python

    # Access remote rank's memory directly
    remote_ptr = dl.symm_at(local_ptr, peer_rank)
    tl.store(remote_ptr + offset, data)  # Write to peer's memory

Kernel Design Patterns
----------------------

AllGather + GEMM Overlapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Time →
    ┌─────────────────────────────────────────────────────────────┐
    │ Producer (AllGather)                                         │
    │ [Tile 0] → [Tile 1] → [Tile 2] → [Tile 3] → ...             │
    │    ↓          ↓          ↓          ↓                        │
    │  signal    signal     signal     signal                      │
    │    ↓          ↓          ↓          ↓                        │
    │ Consumer (GEMM)                                              │
    │         [Tile 0] → [Tile 1] → [Tile 2] → [Tile 3] → ...     │
    └─────────────────────────────────────────────────────────────┘

1. Producer transfers tiles from other ranks via AllGather
2. After each tile transfer completes, producer signals the consumer
3. Consumer waits for each tile, then immediately starts GEMM computation
4. Computation and communication overlap, hiding communication latency

GEMM + ReduceScatter Overlapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Time →
    ┌─────────────────────────────────────────────────────────────┐
    │ Producer (GEMM)                                              │
    │ [Tile 0] → [Tile 1] → [Tile 2] → [Tile 3] → ...             │
    │    ↓          ↓          ↓          ↓                        │
    │  signal    signal     signal     signal                      │
    │    ↓          ↓          ↓          ↓                        │
    │ Consumer (ReduceScatter)                                     │
    │         [Tile 0] → [Tile 1] → [Tile 2] → [Tile 3] → ...     │
    └─────────────────────────────────────────────────────────────┘

1. Producer (GEMM) computes output tiles
2. After each tile is computed, producer signals the consumer
3. Consumer (ReduceScatter) waits for tiles and performs reduction + scatter
4. Communication happens as computation produces results, maximizing overlap

Threadblock Swizzling
~~~~~~~~~~~~~~~~~~~~~

To maximize overlap and minimize synchronization, Triton-distributed uses **threadblock swizzling**:

.. code-block:: python

    # Swizzle tile assignment so each rank starts with its local data
    pid_m = (pid_m + rank * tiles_per_rank) % total_tiles

This ensures:

- Each rank starts computing on locally available data (no wait needed)
- By the time a rank needs remote data, it's likely already transferred
- Reduces synchronization overhead and improves cache efficiency

Token-Based Data Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``consume_token`` primitive establishes explicit data dependencies:

.. code-block:: python

    # Without consume_token: compiler might reorder load before wait
    token = dl.wait(signal_ptr, 1, "gpu", "acquire")
    data = tl.load(data_ptr)  # BUG: might execute before wait!

    # With consume_token: explicit dependency
    token = dl.wait(signal_ptr, 1, "gpu", "acquire")
    data_ptr = dl.consume_token(data_ptr, token)  # Establishes dependency
    data = tl.load(data_ptr)  # Guaranteed to execute after wait

This is essential for correctness because:

1. GPU compilers aggressively reorder instructions
2. The wait and load have no syntactic dependency
3. ``consume_token`` creates an explicit data dependency that prevents reordering

Best Practices
--------------

1. **Minimize Synchronization Granularity**: Use per-tile signals instead of global barriers

2. **Overlap Aggressively**: Start computation as soon as any tile is ready, don't wait for all tiles

3. **Use Threadblock Swizzling**: Arrange work so local data is processed first

4. **Batch Signals When Possible**: Multiple tiles can share a signal if they're always accessed together

5. **Consider Memory Scope**: Use ``"gpu"`` scope for intra-node, ``"sys"`` for inter-node

6. **Profile and Tune**: Use the built-in profiler to identify overlap efficiency

References
----------

- `TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives (MLSys 2025) <https://arxiv.org/abs/2503.20313>`_
- `Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler <https://arxiv.org/abs/2504.19442>`_
