# End-to-End Demo for Triton-Distributed
## Environment Set Up

First, you need to set up the environment for running the end-to-end demo. This includes installing necessary dependencies and configuring the environment variables. You can do this by running the following commands:
```bash
bash ./scripts/build_e2e_env.sh
source ./scripts/setenv.sh
```

## Layer Level End-to-end Demo

We provide TP_MLP, TP_Attn, EP_MoE, SP_Attn for end-to-end demo. You can run the end-to-end demo for these layers by executing the following commands:
```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
# tp mlp
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 4096 --model Qwen/Qwen3-32B --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 128 --model Qwen/Qwen3-32B --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_mlp.py --M 2048 --model Qwen/Qwen3-32B --mode gemm_ar

# tp attn prefill
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 1 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 8 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode gemm_ar

# tp attn decode
bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode allreduce
NVSHMEM_DISABLE_CUDA_VMM=0 bash scripts/launch.sh python/triton_dist/test/nvidia/test_tp_attn.py --bsz 128 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode gemm_ar
```

## Model Level End-to-end Demo

We provide a model level end-to-end demo. You can run the end-to-end demo executing the following command:
```bash
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model Qwen/Qwen3-32B --check --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 32 --seq_len 128 --model Qwen/Qwen3-32B --run_type prefill --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 4096 --seq_len 128 --model Qwen/Qwen3-32B --run_type decode --mode ag_rs
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend torch
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --backend triton_dist
```

## Perf for ByteDance-Seed/Seed-OSS-36B-Instruct on 8xH800:

| Test Case           | Parameters         | Torch AR (s) | Triton Dist AR (s) | Speedup |
| :------------------ | :----------------- | :----------- | :----------------- | :------ |
| **MLP** | `M=2048`           | 0.6587       | 0.4930             | **1.34x** |
| **Attn Prefill** | `bsz=1, ctx=128`   | 0.1274       | 0.0862             | **1.48x** |
| **Attn Decode** | `bsz=128, ctx=128` | 0.1367       | 0.0981             | **1.39x** |
| **E2E Model Prefill** | `bsz=1, ctx=128`   | 15.6478      | 11.8060            | **1.33x** |
| **E2E Model Decode** | `bsz=128, ctx=128` | 16.4576      | 12.4679            | **1.32x** |

