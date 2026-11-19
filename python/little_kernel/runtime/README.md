# Runtime services

This directory owns shared typed ABI, header dependency scanning, immutable
build identity, atomic binary-cache publication and lifecycle helpers. Device
compilation, module loading, launch and synchronization belong to the selected
NVIDIA or AMD backend. Public runtime exports are resolved lazily from installed
descriptors.

CUDA and HIP launches carry the frontend-derived parameter signature. HIP uses
typed contiguous argument storage, honors the selected device and supports
retryable module cleanup. CUDA retains context/module ownership and stream-aware
synchronization. Passing another backend's compiler options raises explicitly.

The SM100 pipeline header is an installed resource at
`little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh`. Header
discovery uses package resources and never reads repository example files.
User headers, target/compiler identity and ordered options participate in cache
identity. See the [repository guide](../../../docs/little_kernel/README.md).
