Intrinsics Reference
====================

Use ``get_backend(backend_id).language`` to obtain a target-scoped language view
containing portable types/helpers together with that architecture's intrinsics,
without importing unrelated backends:

.. code-block:: python

    from little_kernel.backends import get_backend

    sm90_ll = get_backend("nv/sm90").language
    sm100_ll = get_backend("nv/sm100").language
    gfx950_ll = get_backend("amd/cdna4").language

The top-level ``little_kernel.language`` module provides lazy lookup across
installed backend descriptors and raises explicitly if an unbundled symbol is
requested.

.. contents:: On this page
   :local:
   :depth: 2

Portable Operations
-------------------

Shared scalar/tensor/vector types, type casts, ``sizeof``, ``cdiv``, alignment
directives (``align_memory``), conditional selection, and loop annotations
(``unroll``) live under ``little_kernel.language``.

NVIDIA SM90 & SM100 Intrinsics
------------------------------

- **Common CUDA / PTX (``backends.nv.common.language.intrin``)**:

  - Thread/block/cluster coordinates, warp shuffles, ballot/vote, register count
    controls (``setmaxnreg_inc`` / ``setmaxnreg_dec``), ``mbarrier`` initialization,
    transaction byte tracking, phase waits, 1D/2D/multicast ``TMA`` loads & stores,
    ``LDSM`` / ``STSM``, PTX atomics (including 64-bit ``atomic_cas_i64`` and
    FP32/INT16 CAS legalization), packed ``pack_2xbf16``, and round-to-nearest FP32
    math helpers.

- **SM90 (``backends.nv.sm90``)**:

  - Warp-group ``WGMMA`` matrix-multiply descriptors and async pipeline atoms
    (``wgmma.mma_async``, ``wgmma.fence``, ``wgmma.commit_group``,
    ``wgmma.wait_group``) across BF16, FP16, TF32, and FP8 operand layouts.

- **SM100 (``backends.nv.sm100``)**:

  - 1-CTA and 2-CTA ``tcgen05`` (``UMMA``) instructions, SMEM descriptor builders
    (``make_umma_smem_desc`` and dataflow ``encode_umma_smem_desc``), Tensor Memory
    (``TMEM``) allocation/load/store/deallocation, 2-CTA ``TMA`` multicast copies,
    and logical ``dataflow.intrinsics`` (see :doc:`dataflow`).

AMD GFX950 (CDNA4) Intrinsics
-----------------------------

``backends.amd.cdna4.language.intrin.amdgcn`` exposes **GFX950** HIP/AMDGPU
primitives, including:

- Buffer resource descriptor construction and direct global-to-LDS asynchronous
  DMA loads (``buffer_load ... lds``).
- LDS barrier & ``s_waitcnt`` synchronization controls and wave-level scheduling
  barriers.
- Register vector types and block-scaled ``MXFP8`` matrix-core ``MFMA`` intrinsics
  used by the data-parallel and XCD-persistent ``GFX950`` GEMM kernels under
  ``benchmarks/amd/cdna4/``.

Safety Contracts
----------------

Each backend's ``backend.json`` descriptor enumerates its exact public exports.
Typed IR nodes carry result types and memory/synchronization effects consumed by
static analysis, optimization passes, and C++ codegen. Unsupported dtypes,
misaligned shared-memory descriptors, or mismatched launch/architecture flags
fail closed during compilation rather than emitting invalid device code.
