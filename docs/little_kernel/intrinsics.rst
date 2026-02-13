Intrinsics Reference
====================

LittleKernel exposes hardware intrinsics through ``little_kernel.language``
(commonly imported as ``ll``).  Each intrinsic maps to one or more PTX
instructions.

.. contents:: On this page
   :local:
   :depth: 2

Common Intrinsics
-----------------

Thread Indexing
~~~~~~~~~~~~~~~

.. code-block:: python

    ll.threadIdx_x()       # threadIdx.x
    ll.threadIdx_y()       # threadIdx.y
    ll.blockIdx_x()        # blockIdx.x
    ll.blockIdx_y()        # blockIdx.y
    ll.blockDim_x()        # blockDim.x
    ll.get_lane_idx()      # %laneid
    ll.get_warp_idx()      # threadIdx.x / 32

Synchronization
~~~~~~~~~~~~~~~

.. code-block:: python

    ll.syncthreads()       # __syncthreads()
    ll.syncwarp()          # __syncwarp(0xFFFFFFFF)
    ll.threadfence()       # __threadfence()

Memory
~~~~~~

.. code-block:: python

    ll.get_smem_address(buf)     # cvta.shared.u32 (SMEM pointer)
    ll.get_smem_address64(buf)   # cvta.shared.u64

Timing
~~~~~~

.. code-block:: python

    ll.clock64()           # clock64() for latency measurement

SM90 (Hopper) Intrinsics
-------------------------

WGMMA (Warp Group MMA)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ll.wgmma_init_accum_64x64()      # Zero-init m64n64 FP32 accumulator
    ll.wgmma_init_accum_64x128()     # Zero-init m64n128 FP32 accumulator
    ll.wgmma_init_accum_64x256()     # Zero-init m64n256 FP32 accumulator
    ll.wgmma_fence()                 # wgmma.fence.sync.aligned
    ll.wgmma_commit()                # wgmma.commit_group.sync.aligned
    ll.wgmma_wait(n)                 # wgmma.wait_group.sync.aligned N
    ll.wgmma_m64n64k16_bf16(ad, bd)  # WGMMA instruction (descriptor-based)
    ll.wgmma_m64n128k16_bf16(ad, bd)
    ll.wgmma_m64n256k16_bf16(ad, bd)
    ll.wgmma_store_d_64x64(C, ...)   # Store accumulator to global memory
    ll.wgmma_store_d_64x128(C, ...)
    ll.wgmma_store_d_64x256(C, ...)

TMA (Tensor Memory Accelerator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 2D TMA load (global -> shared)
    ll.tma_load_2d(desc, bar, smem_addr, coord_x, coord_y)

    # 2D TMA store (shared -> global)
    ll.tma_store_2d(desc, smem_addr, coord_x, coord_y)
    ll.tma_store_arrive()
    ll.tma_store_wait()

    # 1D bulk copy
    ll.tma_copy_1d_g2s(smem_addr, gmem_addr, bar, size)

MBarrier
~~~~~~~~

.. code-block:: python

    ll.mbarrier_init(bar, count)
    ll.mbarrier_arrive_expect_tx(bar, tx_bytes)
    ll.mbarrier_try_wait_parity(bar, parity)
    ll.mbarrier_arrive(bar)
    ll.mbarrier_invalidate(bar)

Cluster
~~~~~~~

.. code-block:: python

    ll.cluster_arrive()
    ll.cluster_wait()
    ll.cluster_sync()
    ll.get_cluster_ctaid()

SM100 (Blackwell) Intrinsics
-----------------------------

UMMA (Unified MMA / tcgen05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ll.elect_one()                        # elect.sync (returns bool)
    ll.tcgen05_alloc(num_cols)            # TMEM allocation
    ll.tcgen05_dealloc(tmem, num_cols)    # TMEM deallocation
    ll.tcgen05_ld_4x(tmem, smem_addr)    # Load 4 rows from SMEM to TMEM
    ll.tcgen05_ld_8x(tmem, smem_addr)    # Load 8 rows from SMEM to TMEM
    ll.tcgen05_fence_before()             # Fence before MMA
    ll.tcgen05_fence_after()              # Fence after MMA
    ll.umma_m256nNk16_bf16(idesc, tmem, ad, bd)  # UMMA MMA instruction
    ll.umma_commit()                      # Commit UMMA group

TMA (SM100 / cta_group::2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ll.tma_load_2d_cg2(desc, bar, smem_addr, x, y)  # TMA load with PEER_BIT_MASK
    ll.tma_store_2d_sm100(desc, smem_addr, x, y)     # SM100 TMA store
    ll.tma_store_commit()                              # SM100 TMA store commit

TMEM Operations
~~~~~~~~~~~~~~~

.. code-block:: python

    ll.tmem_store_bf16_row(tmem, D, row_off, N, ldm)  # TMEM -> global (BF16)
    ll.tmem_epilogue_coalesced_4w(...)                # Optimized epilogue (requires 4 warps)

Cluster MBarrier (SM100)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ll.mbarrier_arrive_cluster(bar)
    ll.mbarrier_arrive_expect_tx_cluster(bar, tx)

Utilities
~~~~~~~~~

.. code-block:: python

    ll.pack_bf16(a, b)        # Pack two BF16 into uint32
    ll.st_shared_128(addr, v0, v1, v2, v3)  # 128-bit shared store
    ll.uint_as_float(x)       # Reinterpret uint32 as float32
    ll.__shfl_sync(mask, val, src_lane, width)  # Warp shuffle

Runtime Helpers
---------------

TMA Descriptor Creation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor

    desc = create_tma_2d_descriptor(
        tensor,
        gmem_inner_dim=K,
        gmem_outer_dim=M,
        smem_inner_dim=BK,
        smem_outer_dim=BM,
        gmem_outer_stride=K,
        swizzle_mode=128,       # B128 swizzle
        oob_fill=0,
        l2_promotion=3,         # L2_256B
    )
