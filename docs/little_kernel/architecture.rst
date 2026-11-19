Architecture & Compiler Pipeline
================================

LittleKernel separates target-neutral compiler infrastructure from hardware-owned
language extensions, optimization passes, code generators, toolchains, and runtimes.
Only the physically bundled backends—``nv/sm90`` (**SM90**), ``nv/sm100`` (**SM100**),
and ``amd/cdna4`` (**GFX950**)—are registered and discoverable at runtime.

.. contents:: On this page
   :local:
   :depth: 2

Design Principles
-----------------

1. **Explicit Hardware Control in Python**: Kernels are authored as typed Python
   functions decorated with ``@lk.ll_kernel`` and translated through an AST-to-C++
   compiler pipeline rather than traced via Python execution. Standard ``if/else``,
   runtime ``for``/``while`` loops, and loop-carried mutable scalars lower directly
   to structured CUDA C++ or HIP C++ without SSA restrictions.
2. **Strict Backend Isolation**: ``little_kernel.backends.interfaces`` defines
   neutral contracts (``BackendSpec``, ``BackendComponents``, ``BuildRequest``),
   and ``little_kernel.backends.registry`` discovers installed ``backend.json``
   descriptors lazily without importing unrelated hardware stacks.
3. **Zero-GPU Source Generation**: Lowering and C++/CUDA/HIP source emission run
   purely on CPU with Python 3.11+ and NumPy. Device toolchains (``nvcc`` or
   ``hipcc``), drivers, and tensor frameworks (PyTorch) are invoked only when
   compiling binaries via ``.build(...)`` or launching on hardware.
4. **Opt-In High-Level Scheduling Where Needed**: Authors can write explicit
   warp-specialized kernels directly using ``SM90`` / ``SM100`` / ``GFX950``
   intrinsics, or opt into the ``SM100`` role/dataflow frontend
   (``frontend=dataflow_frontend``) for automatic multi-stage pipeline synthesis.

End-to-End Compilation Pipeline
-------------------------------

::

    @lk.ll_kernel Python AST
            │
            ▼
    [Optional Explicit Frontend]
      e.g. SM100 dataflow_frontend (roles, channels, buffer overlay, auto-pipelining)
            │
            ▼
    [Shared & Backend Pass Pipeline — PASSES[backend_id]]
      • Symbol resolution & constexpr folding
      • Type inference (with C++ integral promotion) & static int/alignment analysis
      • Compile-time loop unrolling (ll.unroll) & empty-region flattening
      • Shared/dynamic-shared memory layout & SIMT allocation analysis
      • Backend-specific legalization (e.g. PTX atomics, WGMMA/UMMA/MFMA descriptors)
            │
            ▼
    [Code Generation — get_backend(backend_id).codegen.generate]
      • CUDA C++ / inline PTX emission (nv/sm90, nv/sm100)
      • HIP C++ / inline AMDGPU emission (amd/cdna4 for GFX950)
            │
            ▼
    [Runtime & Content-Addressed Cache — little_kernel.runtime]
      • Toolchain invocation (nvcc / hipcc) keyed by source + arch + flags + headers
      • Typed host ABI marshalling (scalars, pointers, __grid_constant__ TMA descriptors)
      • Stream-aware launch & cluster launch attributes

Component Ownership
-------------------

==================================== ==========================================================
Layer                                Repository Path & Responsibility
==================================== ==========================================================
Shared Core & IR                     ``python/little_kernel/core/`` — types, typed IR effects, compiler sessions, pass manager
Portable Language                    ``python/little_kernel/language/`` — scalar/tensor/vector types, portable casts, loop modifiers
Shared Codegen                       ``python/little_kernel/codegen/`` — generic C++ AST visitors, operator/type registries, struct lowering
Runtime & Cache                      ``python/little_kernel/runtime/`` — ABI marshalling, dependency-aware binary cache, module lifecycle
Backend Registry                     ``python/little_kernel/backends/{interfaces,registry,shared}.py`` — descriptor discovery & lazy factories
NVIDIA Common                        ``python/little_kernel/backends/nv/common/`` — shared CUDA/PTX intrinsics, TMA descriptor helpers, NVCC driver
NVIDIA SM90                          ``python/little_kernel/backends/nv/sm90/`` — ``WGMMA`` atoms, ``mbarrier`` pipelines, ``STSM``, cluster primitives
NVIDIA SM100                         ``python/little_kernel/backends/nv/sm100/`` — ``tcgen05`` ``UMMA``, ``TMEM``, 2-CTA TMA, role/dataflow compiler & ``role_pipeline.cuh``
AMD GFX950 (CDNA4)                   ``python/little_kernel/backends/amd/cdna4/`` — ``amdgcn`` DMA/LDS/MFMA intrinsics, HIP emitter, ``hipcc`` runtime
Tests, Benchmarks & Examples         ``tests/``, ``benchmarks/``, ``examples/`` — hardware-scoped verification & benchmark suites outside the wheel
==================================== ==========================================================

Packaging & Selective Source Export
-----------------------------------

Inside ``Triton-distributed``, ``python/setup.py`` bundles ``little_kernel`` by
default (controlled via ``TRITON_BUILD_LITTLE_KERNEL``), including backend
``backend.json`` descriptors and packaged runtime headers such as
``python/little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh``.

To export a standalone subset containing only selected hardware backends and
their exact dependency closure:

.. code-block:: bash

    uv run --locked python tools/export_source.py \
        --backends nv/sm90,nv/sm100,amd/cdna4 \
        --output /tmp/little-kernel-export
