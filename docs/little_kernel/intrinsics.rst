Intrinsics Reference
====================

Use a target-specific language view to expose shared types and that backend's
primitives without importing another implementation:

.. code-block:: python

    from little_kernel.backends import get_backend
    hopper = get_backend("nv/sm90").language
    blackwell = get_backend("nv/sm100").language
    cdna4 = get_backend("amd/cdna4").language
    print(dir(hopper))

The top-level ``little_kernel.language`` facade retains lazy access to installed
exports. A missing symbol raises explicitly; it does not select an omitted backend.

Shared operations
-----------------

Shared scalar/tensor/vector types, casts, ``sizeof``, ``cdiv``, alignment,
selection and loop annotations live under ``language``. GPU allocation and SIMT
analysis are reusable components configured by each backend's pass provider.

NVIDIA
------

CUDA/PTX intrinsics, common memory operations, barriers and atomics are owned by
``backends.nv.common.language.intrin``. SM90-specific WGMMA and reusable MMA/
pipeline atoms live under ``backends.nv.sm90``. SM100-specific UMMA, two-CTA TMA
and dataflow primitives live under ``backends.nv.sm100``.

The SM100 dataflow frontend is selected explicitly through
``backends.nv.sm100.passes.dataflow.dataflow_frontend``. Its logical resources,
channels and pipeline records have hardware-owned language/core definitions;
analysis/optimization and codegen are separate components. Its required pipeline
header is an installed package resource.

AMD
---

CDNA4 primitives are owned by
``backends.amd.cdna4.language.intrin.amdgcn``. They include AMDGPU memory/resource
operations and MFMA/register-vector support. HIP codegen, toolchain and runtime
components are instantiated independently. The data-parallel and persistent
MXFP8 GEMM programs are under ``benchmarks/amd/cdna4``.

Contracts
---------

Backend descriptors enumerate the actual public exports. Typed operations carry
result/effect information used by type inference, passes and emission. Invalid
type, alignment, architecture and launch options fail rather than producing
unsupported device code. CPU source-generation tests are separate from device
correctness and timing gates.
