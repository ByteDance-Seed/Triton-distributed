# Running Tests

This guide explains how to run tests for Triton-distributed on NVIDIA and AMD GPUs.

## Prerequisites

Before running tests, ensure you have:

1. Completed the [build process](build.md)
2. Set up the environment:

```bash
source ./scripts/setenv.sh
```

## Test Categories

Triton-distributed provides comprehensive tests for all kernels and layers.

| Category | Description |
|----------|-------------|
| Unit Tests | Core kernel functionality tests |
| Tutorial Tests | Tutorial example validation |
| E2E Tests | End-to-end model integration tests |
| Mega Kernel Tests | Mega Triton Kernel tests |

## How to Run Tests

All tests are run using the `scripts/launch.sh` script which handles distributed setup:

```bash
# Basic usage
bash scripts/launch.sh <test_script.py> [args]

# With specific number of GPUs
bash scripts/launch.sh --nproc_per_node=4 <test_script.py> [args]

# With environment variables
NVSHMEM_SYMMETRIC_SIZE=10g bash scripts/launch.sh <test_script.py> [args]
```

---

## NVIDIA GPU Tests

### Language Extensions
```bash
python3 python/triton_dist/test/nvidia/test_language_extra.py
python3 python/triton_dist/test/common/test_language_extra.py
```

### SIMT Operations
```bash
python3 python/triton_dist/test/common/test_simt.py
python3 python/triton_dist/test/common/test_simt_vec_add.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness
bash scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
```

### AllGather + GEMM
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check --autotune
```

### GEMM + ReduceScatter
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 4096 -N 4096 -K 12288 --fuse_scatter --check
```

### AllGather
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_small_msg.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_all_gather.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_fast_allgather.py --iters 10 --warmup_iters 20 --mode push_2d_ll --minbytes 4096 --maxbytes 8192
```

### AllReduce
```bash
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method double_tree --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method two_shot --stress --iters 2 --verify_hang 50
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot_multimem --stress --iters 2 --verify_hang 50
```

### Expert Parallelism All-to-All
```bash
# Standard EP A2A
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check

# With scatter indices and weights
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 4096 -N 6144 --topk 6 --drop_ratio 0.3 --check --with-scatter-indices --has_weight

# With local combine optimization
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 32768 -N 1536 --topk 8 -G 384 --drop_ratio 0.3 --enable-local-combine --check

# Low-latency mode
NVSHMEM_SYMMETRIC_SIZE=2g bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_ll_a2a.py -M 128 --iters 5 --verify-iters 20 --check

# AOT compiled version
NVSHMEM_SYMMETRIC_SIZE=10000000000 bash scripts/launch.sh python/triton_dist/test/nvidia/test_ep_a2a.py -M 8192 -N 7168 --topk 8 --check --use_aot
```

### Flash Decoding
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
bash scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k_persistent
bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
```

### GEMM + AllReduce
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 32 5120 25600 --check --low-latency
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_ar.py 28000 7168 4096 --check --num_comm_sms 4
```

### MoE Kernels
```bash
# AllGather MoE
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20 --autotune

# MoE ReduceScatter
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 14336 4096 64 4

# MoE AllReduce
bash scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_ar.py 8192 2048 1536 32 2
```

### Sequence Parallel Attention
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_sp_ag_attention_intra_node.py --batch_size 1 --q_head 32 --kv_head 32 --max_seqlen_q 8192 --max_seqlen_k 8192 --head_dim 128 --seqlens_q 8192 --seqlens_k 8192
```

### Ulysses Sequence Parallelism
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ulysses_sp_dispatch.py 1 8000 32 128 --gqa 8 --check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ulysses_sp_dispatch.py 1 16384 8 128 --gqa 8
```

### NVSHMEM API
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_ring_put.py
bash scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_init.py
```

### AOT Compilation
```bash
USE_TRITON_DISTRIBUTED_AOT=1 bash scripts/launch.sh python/triton_dist/test/nvidia/test_compile_aot.py
```

---

## Tutorial Tests

Run all tutorial examples to verify your installation:

```bash
bash scripts/launch.sh tutorials/01-distributed-notify-wait.py
bash scripts/launch.sh tutorials/02-intra-node-allgather.py
bash scripts/launch.sh tutorials/03-inter-node-allgather.py
bash scripts/launch.sh tutorials/04-deepseek-infer-all2all.py
bash scripts/launch.sh tutorials/05-intra-node-reduce-scatter.py
bash scripts/launch.sh tutorials/06-inter-node-reduce-scatter.py
bash scripts/launch.sh tutorials/07-overlapping-allgather-gemm.py
bash scripts/launch.sh tutorials/08-overlapping-gemm-reduce-scatter.py
```

---

## E2E Model Tests

### Dense Model Tests
```bash
# TP MLP
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model <model_path> --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model <model_path> --mode allreduce

# TP Attention (Prefill)
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model <model_path> --run_type prefill --mode ag_rs

# TP Attention (Decode)
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model <model_path> --run_type decode --mode ag_rs

# TP E2E Check
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <model_path> --check --mode ag_rs

# Full Inference
bash scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist
```

### MoE Model Tests
```bash
bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_moe.py --bsz 32 --seq_len 128 --model <moe_model_path>
bash scripts/launch.sh --nproc_per_node=4 python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <moe_model_path> --check --mode ag_rs
```

### Pipeline Parallelism Tests
```bash
bash scripts/launch.sh python/triton_dist/test/nvidia/test_pp_block.py --bsz 8 --seq_len 128 --num_blocks 4 --pp_size 4 --model <model_path>
```

---

## Mega Triton Kernel Tests

```bash
# Individual ops
python3 python/triton_dist/mega_triton_kernel/test/ops/test_attn_layer.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_mlp_layer.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_rms_norm.py
python3 python/triton_dist/mega_triton_kernel/test/ops/test_flash_attn.py

# AllReduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/ops/test_allreduce.py

# Full model test
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/test_qwen3.py --model <qwen_model_path> --backend mega_kernel

# Benchmark
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model <qwen_model_path> --seq_len 128 --allreduce_method one_shot_multimem
```

---

## AMD GPU Tests

### GEMM ReduceScatter
```bash
bash scripts/launch_amd.sh python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
```

### AMD Tutorials
```bash
bash scripts/launch_amd.sh tutorials/09-AMD-overlapping-allgather-gemm.py
bash scripts/launch_amd.sh tutorials/10-AMD-overlapping-gemm-reduce-scatter.py
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NVSHMEM_SYMMETRIC_SIZE` | Symmetric heap size | `10000000000` or `2g` |
| `NVSHMEM_DISABLE_CUDA_VMM` | Disable CUDA VMM | `0` or `1` |
| `NVSHMEM_IBGDA_SUPPORT` | Enable IB GDA support | `1` |
| `USE_TRITON_DISTRIBUTED_AOT` | Use AOT compiled kernels | `1` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Max CUDA connections | `8` |

---

## Troubleshooting

### Common Issues

1. **NVSHMEM initialization failure**:
   - Increase `NVSHMEM_SYMMETRIC_SIZE`
   - Set `NVSHMEM_DISABLE_CUDA_VMM=0` or `=1` depending on your system

2. **Hang during allreduce**:
   - Add `--verify_hang` flag with a timeout value
   - Check NVLink/IB connectivity

3. **OOM errors**:
   - Reduce batch size or sequence length
   - Use smaller hidden dimensions

4. **Test timeouts**:
   - Reduce `--iters` count
   - Check GPU utilization for bottlenecks

## See Also

- [Build Instructions](build.md)
- [Tutorials](getting-started/tutorials/index)
- [E2E Integration](getting-started/e2e/index)
