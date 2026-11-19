SM100 Role & Dataflow Compiler
==============================

On **SM100** (``nv/sm100``), asynchronous tensor engines—``TMA`` bulk transfers,
1-CTA / 2-CTA ``tcgen05`` (``UMMA``) matrix-multiply units, Tensor Memory
(``TMEM``), and ``TMA`` bulk stores—operate concurrently across specialized warps
with distinct completion proxies.

LittleKernel provides an explicit **Role & Dataflow frontend**
(``little_kernel.backends.nv.sm100.passes.dataflow.dataflow_frontend``) that lets
kernel authors declare hardware roles, staged logical memories, and producer-consumer
data channels in concise Python while the compiler synthesizes barrier allocation,
phase tracking, CTA-group ``tcgen05.commit`` arrivals, and ``TMA`` bulk-group
retirement automatically.

.. contents:: On this page
   :local:
   :depth: 2

Programming Model
-----------------

Select ``frontend=dataflow_frontend`` on ``@lk.ll_kernel(backend="nv/sm100", ...)``
and declare the kernel's role graph with ``with dataflow() as g:``:

.. code-block:: python

    import little_kernel as lk
    import little_kernel.language as ll
    from little_kernel.backends import get_backend
    from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
    from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
    from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend
    from little_kernel.core.passes import PASSES

    @lk.ll_kernel(backend="nv/sm100", is_entry=True, frontend=dataflow_frontend)
    def pipelined_gemm_sketch(M: ll.int32, N: ll.int32, K: ll.int32) -> ll.void:
        with dataflow() as g:
            tma_role = g.role("tma_load", hw=HW.TMA, warps=[0])
            mma_role = g.role("umma", hw=HW.TensorCore, warps=[1])
            epi_role = g.role("epilogue", hw=HW.CUDA, warps=[4, 5, 6, 7])

            smem_a = g.smem("smem_a", [128, 64], dtype=ll.bfloat16, stages=4)
            smem_b = g.smem("smem_b", [128, 64], dtype=ll.bfloat16, stages=4)
            tmem_c = g.tmem("tmem_c", [128, 256], dtype=ll.float32, stages=2)

            ab_pipe = g.depends(src=tma_role, dst=mma_role, via=[smem_a, smem_b])
            acc_pipe = g.depends(src=mma_role, dst=epi_role, via=[tmem_c])

        # Each role writes sequential tile logic and binds logical stage keys
        # (e.g., ab_pipe.bind(k_step)); the compiler inserts producer/consumer
        # waits, mbarrier byte transactions, and tcgen05 commits.

Core Concepts
-------------

- **Roles (``g.role``)**: Assign disjoint warp groups to hardware engines
  (``HW.TMA``, ``HW.TensorCore``, ``HW.CUDA``).
- **Logical Staged Memories (``g.smem``, ``g.tmem``)**: Declare multi-stage
  shared-memory tiles and ``SM100`` Tensor Memory accumulators without manual
  stage-modulo indexing or byte-offset arithmetic.
- **Channels (``g.depends``) vs. Lowered Pipelines**:

  - A **channel** declares a source-level logical RAW dependency (supporting
    ``SPSC``, ``SPMC``, and ``MPMC`` producer/consumer topologies) with
    ``stage.bind(step)`` markers.
  - A **pipeline** is the lowered runtime synchronization object materialized by
    the compiler into ``role_pipeline.cuh`` primitives.

- **Physical Storage Overlay (``MemoryMappingEntry``)**: Multiple non-overlapping
  logical views (such as mainloop operand buffers and epilogue ``TMA`` staging
  buffers) can share physical dynamic shared memory; the overlay pass proves
  lifetime safety and inserts internal reuse barriers automatically.

Ten-Stage Dataflow Pass Pipeline
--------------------------------

When ``PASSES["nv/sm100"]`` runs after ``dataflow_frontend``, the dataflow
compiler executes ten fail-closed analysis and lowering stages:

1. **Compile-time loop unrolling** for static inner tile loops.
2. **UMMA mode & effect normalization** (separating first-K initialization from accumulation).
3. **Channel, ownership, bind-stream, and TMA-partition validation** (rejecting mismatched producer/consumer loop streams at compile time).
4. **Structured control-flow partitioning** at logical memory effect frontiers.
5. **Loop-carried RMW-cycle analysis**.
6. **Physical-buffer overlay lifetime analysis**.
7. **Inferred overlay-pipeline construction**.
8. **Per-domain static TMA-store bulk-group stream scheduling** (emitting literal ``wait_group.read<N>`` retirements with zero runtime group tracking overhead).
9. **Pipeline synchronization primitive insertion** across role branches.
10. **CUDA C++ / PTX resource, descriptor, and ``role_pipeline.cuh`` materialization**.

Bundled SM100 Dataflow Kernels
------------------------------

Complete single-GPU and distributed multi-GPU **SM100** kernels built with the
dataflow compiler are provided in ``examples/nv/sm100/`` and benchmarked in
``benchmarks/nv/sm100/dataflow/``:

- **Dense & Block-Scaled 2-CTA UMMA GEMMs**: ``gemm_sm100.py`` (BF16), ``mxfp8_gemm.py`` (MXFP8), ``mxfp4_gemm.py`` (MXFP4), ``nvfp4_gemm.py`` (NVFP4), ``grouped_gemm.py`` (BF16 Grouped GEMM), ``mxfp_w4a8_groupgemm.py`` (W4A8 MXFP Grouped GEMM).
- **Attention & Sequence Modeling**: ``fa4_attention.py`` (FlashAttention-4 forward), ``gated_delta_net.py`` (Gated DeltaNet chunk recurrence).
- **Distributed Compute-Communication Overlap**: ``ag_gemm_fused_sm100.py`` (AllGather + GEMM), ``gemm_rs_fused_sm100.py`` (GEMM + ReduceScatter), ``gemm_a2a_fused_sm100.py`` (GEMM + AllToAll), ``a2a_gemm_fused_sm100.py`` (AllToAll + GEMM), and ``dispatch_group_gemm_swiglu_fused_sm100.py`` (Fused EP Dispatch + GroupGEMM + SwiGLU).
