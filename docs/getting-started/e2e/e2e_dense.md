# Dense Model

This document provides an end-to-end (E2E) integration for Triton-Distributed. It is designed to showcase how to integrate Triton-Distributed's high-performance distributed kernels into a complete LLM, using Qwen3-32B as a reference example. The demo covers the tensor parallel implementation and performance testing from individual layers (Attention, MLP) to the entire model.

## Features

  * **Two Strategies for Tensor Parallelism (TP)**:
      * Utilizes `AllGather-GEMM` and `GEMM-ReduceScatter` kernels. The input is sharded along the `batch` dimension, and communication is highly overlapped with computation.
      * Employs `GEMM + AllReduce`. The input is replicated across all devices.
  * **Layer-wise Module Implementation**: Provides `TP_Attn` and `TP_MLP` modules that can easily replace corresponding layers in existing models to enable distributed parallelism.
  * **Full Model Integration**: Demonstrates how to seamlessly integrate the parallel modules into a dense model, using `Qwen3-32B` as an example. We also include a complete inference `Engine` with CUDA Graph integration.

**Perf on 8xH800:** Large tensor shapes are best suited for a pipelined `AllGather-GEMM + GEMM-ReduceScatter` to overlap computation and communication, while smaller shapes are more efficient with `GEMM + AllReduce` .

- `AllGather-GEMM` + `GEMM-ReduceScatter`

| Test Case | Parameters | Torch AR (ms) | Dist-Triton (ms) | Speedup |
|---|---|---|---|---|
| **MLP** | `M=4096` | 1.076972 | 0.8854406 | **1.216** |
| **Attn Prefill** | `bsz=32, ctx=128` | 0.71913 | 0.748670 | 0.961* |
| **Attn Decode** | `bsz=4096, ctx=128` | 1.29802 | 1.31813 | 0.985* |
| **E2E Model Prefill**| `bsz=32, ctx=128` | 123.3569 | 104.2794 | **1.183** |
| **E2E Model Decode**| `bsz=4096, ctx=128` | 160.1424 | 140.393 | **1.141** |

*The items marked with an asterisk show negative performance gains (i.e., slower speeds). This is because the shape of the weight tensors in the Attention computations is very small. For small-sized tensors, the additional overhead of splitting the communication operation into AllGather and ReduceScatter outweighs the gains from overlapping the computations, so the performance is worse than PyTorch's single AllReduce operation.

- `GEMM + AllReduce`

**Note:** The AllReduce implementation is not optimized yet in this open-source codebase, and will be optimized later.

| Test Case | Parameters | Torch AR (ms) | Triton Dist AR (ms) | Speedup |
|---|---|---|---|---|
| **MLP** | `M=128` | 0.1255 | 0.0918 | **1.37x** |
| **Attn Prefill** | `bsz=1, ctx=128` | 0.1275 | 0.0970 | **1.31x** |
| **Attn Decode** | `bsz=128, ctx=128` | 0.1438 | 0.113 | **1.27x** |
| **E2E Model Prefill** | `bsz=1, ctx=128` | 15.97 | 12.30 | **1.30x** |
| **E2E Model Decode** | `bsz=128, ctx=128` | 16.68 | 13.003 | **1.28x** |

**Perf on 8xMI308X:** 

| Test Case | Parameters | Torch AR (ms) | Dist-Triton (ms) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **AG_GEMM** | `M=4096` | 1.8047 | 1.8002 | **1.0025x** |
| **GEMM_RS** | `M=4096` | 1.057 | 0.837 | **1.2627x** |
| **MLP** | `M=4096` | 3.019 | 2.829 | **1.067x** |
| **Attn Prefill** | `bsz=32, ctx=128` | 1.555 | 1.50833 | **1.0312x** |
| **Attn Decode** | `bsz=4096, ctx=128`| 3.3783 | 3.2765 | **1.0310x** |

-----

## Environment Setup

First, run the following scripts to install the necessary dependencies and configure your environment variables.

```bash
# Build the environment and install dependencies
bash ./scripts/build_e2e_env.sh
```

-----

## Running the Demos

We provide a set of test scripts for various use cases.

### 1\. Layer-Level Benchmarks

These scripts are used to benchmark the performance of the `TP_Attn` and `TP_MLP` layers in isolation.

#### MLP Layer (`test_tp_mlp.py`)

**AG_GEMM + GEMM_RS Mode**:
This command benchmarks the performance of `ag_gemm` + `gemm_rs`. The input tensor `x`'s `M` dimension (`batch_size * seq_len`) is sharded across GPUs.

```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B
```

**AllReduce Mode**:
Use the `--use_allreduce` flag to switch to the `GEMM + AllReduce` paradigm. In this mode, the input is replicated on all GPUs.

```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model Qwen/Qwen3-32B --use_allreduce --allreduce_method two_shot_multimem
```

#### Attention Layer (`test_tp_attn.py`)

The Attention layer benchmark is divided into `prefill` and `decode` modes.

**AG_GEMM + GEMM_RS Mode**:

```bash
# prefill
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill

# decode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode
```

**AllReduce Mode**:

```bash
# prefill
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill --use_allreduce --allreduce_method two_shot_multimem

# decode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --mode decode --use_allreduce --allreduce_method two_shot_multimem
```

### 2\. Model-Level End-to-End Tests (`test_tp_e2e.py`)

This script tests a single forward pass of the complete Qwen3 model, which can be used for correctness validation or performance evaluation.

**Correctness Check (`--check`)**:
This mode compares the output of the Triton-Distributed implementation against the native PyTorch eager mode implementation to ensure numerical consistency.

```bash
# AG_GEMM + GEMM_RS Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check

# AllReduce Mode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --check --use_allreduce --allreduce_method two_shot_multimem
```

**Performance Benchmark (`--mode`)**:
This mode benchmarks the model's forward pass performance during the `prefill` and `decode` stages.

```bash
# AG_GEMM + GEMM_RS Mode
# Prefill
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill

# Decode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --mode decode

# AllReduce Mode
# Prefill
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --mode prefill --use_allreduce --allreduce_method two_shot_multimem

# Decode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --mode decode --use_allreduce --allreduce_method two_shot_multimem
```


### 3\. Full Inference Pipeline (`test_e2e_inference.py`)

This script runs a complete generation task (including one prefill step and multiple decode steps) using the `Engine` class. It measures end-to-end throughput and latency.
```bash
# Baseline PyTorch Eager Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150

bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150

# Triton-Distributed AG_GEMM + GEMM_RS Mode
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --triton_dist

# Triton-Distributed AllReduce Mode
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 128 --gen_len 128 --max_length 150 --triton_dist_AR
```