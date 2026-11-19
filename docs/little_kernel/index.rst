LittleKernel
============

LittleKernel is a Python DSL and compiler for NVIDIA CUDA/PTX and AMD HIP/AMDGPU
programs. The bundled backends are ``nv/sm90``, ``nv/sm100`` and ``amd/cdna4``.

Shared types, IR, compiler sessions, analyses, C++ visitors, ABI and binary-cache
services are composed with hardware-owned frontend extensions, passes, codegen,
toolchains and runtimes. Discovery reads installed backend descriptors without
importing all hardware implementations.

Architecture
------------

::

    python/little_kernel/
      core/                shared types, IR, sessions and passes
      language/            shared DSL and portable operations
      codegen/             shared visitors and emission policy
      runtime/             shared ABI/cache/lifecycle services
      backends/
        interfaces.py      component contracts and build requests
        registry.py        descriptor discovery and lazy factories
        common/gpu/        reusable GPU allocation/SIMT components
        nv/common/         CUDA/PTX, runtime and export integrations
        nv/sm90/           Hopper WGMMA and pipeline atoms
        nv/sm100/          Blackwell UMMA, dataflow and runtime header
        amd/cdna4/         gfx950 frontend, HIP codegen and runtime

Tests, authored programs, benchmarks and reference assets are repository-level
``tests/``, ``examples/``, ``benchmarks/`` and ``assets/`` trees. They are not
part of the installed compiler package.

Sub-pages
---------

.. toctree::
   :maxdepth: 1

   getting-started
   dsl-reference
   intrinsics
