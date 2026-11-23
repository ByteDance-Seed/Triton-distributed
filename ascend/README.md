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

## 🎯 Introduction

Triton-distributed's ascend module is a distributed computing framework extended from
the Triton language and optimized specifically for Ascend AI processors. It
enables efficient distributed computing on Ascend NPUs, supports a variety of
communication primitives and computation-communication overlap optimizations,
and aims to improve the performance of large-scale distributed training and
inference.

The project leverages the multi-level abstraction capabilities of MLIR
(Multi-Level Intermediate Representation) to provide comprehensive
expressiveness for Ascend hardware. Through compiler optimizations,
distributed operators written in Triton are efficiently mapped to Ascend AI
processors. Fine-grained performance control interfaces are also provided for
precise control over on-chip memory, pipeline synchronization, and more.

(docs: [triton-distributed-ascend.pages.dev](https://triton-distributed-ascend.pages.dev)).

__List of core maintainers (alphabetical order):__
* hailijuan
* weilinquan

### Key Features

- **Computation-communication overlap**: Overlapped execution of computation
  and communication for improved overall performance
- **AscendNPU IR-based compiler optimizations**: In-depth optimizations
  leveraging Ascend hardware features
- **Python-friendly interfaces**: Concise Python APIs that are easy to
  integrate into existing deep learning frameworks

## 📝 Requirements

### Software Requirements

- **CANN**: CANN 9.1.0 or later
  - See the [CANN Community Edition documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=local)
- **PyTorch**: PyTorch 2.0 or later
- **torch_npu**: Select the torch_npu plugin corresponding to your PyTorch
  version
  - See the [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) documentation
- **Toolchain**:
  - CMake ≥ 3.19
  - GLIBC ≥ 2.28
- **Python**: Python 3.8 or later

## ⚡️ Quick Start

The steps below describe the canonical Ascend build flow. In *this* repository,
check out the [`triton-v3.4`](https://github.com/ByteDance-Seed/Triton-distributed/tree/triton-v3.4)
branch firs and refer to [`build workflow`](https://github.com/ByteDance-Seed/Triton-distributed/blob/triton-v3.4/docs/build.md).

## 🙏 Acknowledgments

- [Triton](https://github.com/openai/triton) - OpenAI's Triton language
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR) - Ascend NPU intermediate representation
- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
