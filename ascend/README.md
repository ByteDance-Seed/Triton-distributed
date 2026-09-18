# Ascend sources (parked)

This directory holds the Triton 3.4-era Ascend backend for Triton-distributed.
It is **not** part of the default Triton 3.7 plugin build on `main`.

`triton-ascend` is still based on Triton 3.6 and does not yet expose the
Triton 3.7 plugin API (`TRITON_PLUGIN_PATHS`). Until that rebase exists, a
working Ascend build lives on the [`triton-v3.4`](https://github.com/ByteDance-Seed/Triton-distributed/tree/triton-v3.4) branch.

Layout (paths relative to this directory match the old in-tree locations):

- `python/src/ascend_passes.cc` — pybind wrapper for `convert_triton_distributed_to_hivm`
- `include/TritonDistributed/Conversion/TritonDistributedToHIVM/`
- `lib/Conversion/TritonDistributedToHIVM/`
- `python/triton_dist/language/extra/ascend/`
- `python/triton_dist/test/ascend/`
- `tutorials/ascend/`
- `3rdparty/triton-ascend.patch`, `3rdparty/AscendNPU-IR.patch`

The `3rdparty/triton-ascend` and `3rdparty/shmem` gitlinks stay at the repo
root (`update = none` in `.gitmodules`).

When a plugin-based Ascend port is ready, these sources can be lifted back
into `plugin/` / `python/` rather than rewritten from scratch.
