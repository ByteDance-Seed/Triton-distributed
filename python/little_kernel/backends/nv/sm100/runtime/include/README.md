# Role/dataflow backend headers

`role_pipeline.cuh` is the canonical packaged SM100 pipeline implementation.
It is the only maintained copy; no prototype forwarding header is retained.

Generated CUDA includes `little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh`.
`runtime.compiler.default_include_paths()` supplies the import-package root,
so the same include works in a source checkout and in an installed wheel.
The header includes `<cuda.h>` explicitly for `CUtensorMap`.

SM100 compilation requires a matching CUDA toolkit (12.8 or newer); the
refactor acceptance used NVCC 13.0 with an explicit `compute_100a` / `sm_100a`
code-generation target. Compilation does not imply SM100 numerical or
performance validation. See `docs/refactor-acceptance.md` for measured gates.
