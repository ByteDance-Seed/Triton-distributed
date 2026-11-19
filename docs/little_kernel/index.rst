LittleKernel Compiler & DSL
===========================

**LittleKernel** is the flagship next-generation Python GPU kernel DSL and compiler
in ``Triton-distributed``. It lowers typed Python kernel functions directly into
optimized NVIDIA CUDA/PTX and AMD HIP/AMDGPU programs, giving developers explicit
warp-, instruction-, and barrier-level control without leaving Python.

LittleKernel ships three isolated, first-class hardware backends:

==================== ==================== =============================================================================
Backend ID           Target Architecture  Key Hardware Capabilities
==================== ==================== =============================================================================
``nv/sm90``          **SM90**             Warp-group ``WGMMA``, ``TMA`` 2D/multicast async copies, ``mbarrier`` pipelines, ``STSM`` swizzled epilogues, cluster launch, distributed symmetric memory
``nv/sm100``         **SM100**            1-CTA / 2-CTA ``tcgen05`` (``UMMA``), Tensor Memory (``TMEM``), block-scaled ``MXFP8`` / ``MXFP4`` / ``NVFP4``, 2-CTA multicast ``TMA``, **Role & Dataflow** automatic pipeline compiler
``amd/cdna4``        **GFX950**           Global-to-LDS DMA (``buffer_load ... lds``), XOR-swizzled LDS staging, block-scaled ``MXFP8`` ``MFMA``, XCD-aware persistent tile scheduling
==================== ==================== =============================================================================

Why LittleKernel?
-----------------

- **Zero-Abstraction Ceiling**: Unlike tile-only DSLs that hide warp specialization
  or shared-memory swizzling behind fixed compiler heuristics, LittleKernel exposes
  hardware tensor instructions (``WGMMA``, ``UMMA``, ``MFMA``), asynchronous copy
  engines (``TMA``, global-to-LDS DMA), and barrier transactions as typed Python
  primitives.
- **Structured Control Flow & Loop-Carried Scalars**: Functions decorated with
  ``@lk.ll_kernel`` are compiled through an AST-to-C++ pass pipeline rather than
  traced by Python execution. Runtime ``for``/``while`` loops and mutable
  loop-carried scalars lower directly to native C++ variables for clean software
  pipelining and prefetching.
- **Automatic SM100 Multi-Role Pipelining**: On **SM100**, authors can opt into
  ``frontend=dataflow_frontend`` to declare warp roles (``HW.TMA``,
  ``HW.TensorCore``, ``HW.CUDA``), staged ``SMEM``/``TMEM`` buffers, and logical
  channels—letting the compiler synthesize ``mbarrier`` arrivals, ``tcgen05.commit``
  tracking, physical buffer overlays, and ``TMA`` bulk-group retirements.
- **Decoupled CPU Codegen & Modular Distribution**: Source generation runs on CPU
  with only Python 3.11+ and NumPy; backend discovery inspects lightweight
  ``backend.json`` descriptors without importing unselected vendor toolchains.

Repository Layout
-----------------

::

    python/little_kernel/
      core/                  shared types, typed IR, sessions, and pass manager
      language/              shared DSL types, portable intrinsics, lazy routing
      codegen/               shared C++ visitors and emission-policy registries
      runtime/               typed host ABI, content-addressed cache, lifecycle
      backends/
        interfaces.py        component contracts and build requests
        registry.py          descriptor discovery and lazy backend factories
        common/gpu/          reusable GPU allocation and SIMT analyses
        nv/common/           shared CUDA/PTX intrinsics, TMA descriptors, NVCC runtime
        nv/sm90/             SM90 WGMMA, mbarrier pipelines, and cluster primitives
        nv/sm100/            SM100 UMMA, TMEM, dataflow compiler, and role_pipeline.cuh
        amd/cdna4/           GFX950 frontend, HIP codegen, and hipcc runtime
    benchmarks/{nv,amd}/     SM90, SM100, and GFX950 compute & distributed benchmarks
    examples/nv/             authored SM90 and SM100 compute & overlap programs
    tests/                   CPU codegen, contract, and GPU numerical verification suites

Documentation Guide
-------------------

.. toctree::
   :maxdepth: 2

   getting-started
   architecture
   dsl-reference
   dataflow
   intrinsics
   performance
   validation

.. toctree::
   :hidden:

   README
