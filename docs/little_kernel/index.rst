LittleKernel
============

LittleKernel is a Python DSL for writing CUDA kernels with full PTX-level
control.  It generates CUDA source code from Python functions and compiles
them at runtime using ``nvcc``.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

LittleKernel bridges the gap between high-level frameworks (PyTorch, Triton)
and hand-written CUDA/PTX.  You write kernels as decorated Python functions
using explicit types and intrinsics, and LittleKernel:

1. **Parses** the Python AST,
2. **Runs compiler passes** (type inference, memory allocation, constant
   folding, inlining),
3. **Generates** CUDA C++ source with inline PTX assembly,
4. **Compiles** via ``nvcc`` and wraps the result in a callable that
   interoperates with PyTorch tensors.

Architecture
------------

::

    python/little_kernel/
    ├── language/        # DSL types, decorators, intrinsics
    │   └── intrin/      # Per-feature intrinsic modules
    │       ├── wgmma.py     # SM90 Tensor Core (WGMMA)
    │       ├── umma.py      # SM100 Tensor Core (UMMA / tcgen05)
    │       ├── tma.py       # Tensor Memory Accelerator
    │       ├── barrier.py   # MBarrier & cluster sync
    │       └── ...
    ├── core/            # IR, passes, type system
    │   └── passes/      # Compiler pipeline
    ├── codegen/         # CUDA code generation
    ├── runtime/         # nvcc compilation, TMA descriptors, kernel launch
    ├── atom/            # High-level building blocks (MMA, TMA, barriers)
    └── benchmark/       # GPU micro-benchmarks and GEMM kernels

Supported Architectures
-----------------------

- **SM90 (Hopper)** -- WGMMA, TMA, MBarrier, Cluster, async pipelines
- **SM100 (Blackwell)** -- UMMA, TMEM, tcgen05, 2SM CTA groups, TMA Store

Sub-pages
---------

.. toctree::
   :maxdepth: 1

   getting-started
   dsl-reference
   intrinsics
