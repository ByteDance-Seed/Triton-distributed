# LittleKernel

A Python DSL for writing CUDA kernels with PTX-level control. LittleKernel
generates CUDA C++ with inline PTX from decorated Python functions and
compiles them at runtime via `nvcc`.

## Quick Start

```python
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
import torch

@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(
    A: ll.Tensor[ll.float32],
    B: ll.Tensor[ll.float32],
    C: ll.Tensor[ll.float32],
    N: ll.int32,
) -> ll.void:
    tid: ll.int32 = ll.blockIdx_x() * ll.blockDim_x() + ll.threadIdx_x()
    if tid < N:
        C[tid] = A[tid] + B[tid]

N = 1024
kernel = vec_add.build(PASSES["cuda"], codegen_cuda,
                       grid=(N // 256,), block=(256,), arch="sm_90")

A = torch.randn(N, device="cuda")
B = torch.randn(N, device="cuda")
C = torch.empty(N, device="cuda")
kernel(A, B, C, N)
```

## Architecture

```
little_kernel/
├── language/        # DSL: types, decorators, intrinsic definitions
│   └── intrin/      # Hardware intrinsics (WGMMA, UMMA, TMA, barriers, ...)
├── core/            # IR representation, compiler passes, type system
│   └── passes/      # Type inference, memory allocation, constant folding, inlining
├── codegen/         # CUDA C++ code generation from IR
├── runtime/         # nvcc compilation, TMA descriptor creation, kernel launch
├── atom/            # High-level composable building blocks
└── benchmark/       # GPU micro-benchmarks and GEMM kernel implementations
    ├── gemm_sm90/   # 10 Hopper GEMM variants (v1-v10)
    ├── gemm_sm100/  # 9 Blackwell GEMM levels (level1-level9)
    ├── compute/     # Throughput: FP32 FMA, SFU, WGMMA
    ├── memory/      # Bandwidth: HBM, L1, L2, SMEM, register file
    ├── latency/     # Instruction latency: arithmetic, memory, sync
    ├── warp/        # Shuffle, vote/ballot
    ├── sm/          # Occupancy, IPC
    └── sm90/        # TMA, cluster, async overlap
```

## Supported GPU Architectures

| Architecture | GPU       | Key Features                                |
|-------------|-----------|---------------------------------------------|
| SM90        | H100/H800 | WGMMA, TMA, MBarrier, Cluster, async pipes  |
| SM100       | GB200     | UMMA, TMEM, tcgen05, 2SM CTA groups         |

## Running Benchmarks

```bash
# Full micro-benchmark suite (Hopper)
python -m little_kernel.benchmark.run_all

# Individual GEMM variants
python -m little_kernel.benchmark.gemm_sm90.gemm_v1
python -m little_kernel.benchmark.gemm_sm100.gemm_level1

# All SM100 levels
python -m little_kernel.benchmark.gemm_sm100.test_all_levels
```

## Running Tests

```bash
# Unit tests
pytest test/little_kernel/unit/ -v

# Integration tests (requires GPU)
pytest test/little_kernel/integration/ -v
```
