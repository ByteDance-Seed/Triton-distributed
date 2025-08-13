# Mega Triton Kernel Demo
## Environment Set Up

First, you need to set up the environment. This is exactly the same as the e2e demo. If you have already set up your environment, you can skip this step.
```bash
bash ./scripts/build_e2e_env.sh
source ./scripts/setenv.sh
```


## Chat Demo
We provide a chat demo. You can play with the mega triton kernel using the following command:
```bash
# server
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/model_server.py --model Qwen/Qwen3-32B

# client
python3 python/triton_dist/mega_triton_kernel/test/models/chat.py
```

## Benchmark
We provide a script to benchmark decode latency. If you need to change the TP (Tensor Parallelism) size, you can pass the `nproc_per_node` parameter to the `launch.sh` script.
```bash
NVSHMEM_DISABLE_CUDA_VMM=0 bash ./scripts/launch.sh python/triton_dist/mega_triton_kernel/test/models/bench_qwen3.py --model Qwen/Qwen3-32B --seq_len 512
```


## Build Model
We use Qwen3 as an example(`python/triton_dist/mega_triton_kernel/models/qwen3.py`) to demonstrate how to build a mega triton kernel for a model. You can refer to it when building other models.
