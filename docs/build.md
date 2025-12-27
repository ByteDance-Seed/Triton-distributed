# Build Triton-distributed

## The best practice to use Triton-distributed with the Nvidia backend:
- Python >=3.11 (suggest using virtual environment)
- CUDA >=12.4
- Torch >=2.8

We recommend installation in [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

#### if for AMD GPU:
- ROCM 6.3.0
- Torch 2.4.1 with ROCM support



Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

### NVIDIA Build Steps
1. Prepare docker container:
    ```sh
    docker run --name triton-dist --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -itd nvcr.io/nvidia/pytorch:25.04-py3 /bin/bash
    docker exec -it triton-dist /bin/bash
    ```

2. Clone Triton-distributed to your own path (e.g., `/workspace/Triton-distributed`)
    ```sh
    git clone https://github.com/ByteDance-Seed/Triton-distributed.git
    ```

3. Update submodules
    ```sh
    cd /workspace/Triton-distributed
    git submodule deinit --all -f # deinit previous submodules
    rm -rf 3rdparty/triton # remove previous triton
    git submodule update --init --recursive
    ```

4. Install dependencies (optional for PyTorch container)
    > Note: Not needed for PyTorch container
    ```sh
    # If you are not using PyTorch container
    pip3 install torch==2.8
    pip3 install setuptools==69.0.0 wheel pybind11
    ```

5. Build Triton-distributed

    Then you can build Triton-distributed.

    ```sh
    # Remove triton installed with torch
    pip uninstall triton
    pip uninstall triton_dist # remove previous triton-dist
    # Install dependencies
    pip3 install cuda.core==0.2.0 cuda-python==12.4 nvidia-nvshmem-cu12==3.3.9 Cython==0.29.24 nvshmem4py-cu12==0.1.2
    rm -rf /usr/local/lib/python3.12/dist-packages/triton
    # Install Triton-distributed
    cd /workspace/Triton-distributed
    export USE_TRITON_DISTRIBUTED_AOT=0
    echo 'numpy<2' > /tmp/pip_install_constraint.txt
    MAX_JOBS=126 pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --no-build-isolation --use-pep517
    ```

    We also provide AOT version of Triton-distributed. If you want to use AOT (**Not Recommended**), then
    ```sh
    cd /workspace/Triton-distributed/
    bash ./scripts/gen_aot_code.sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    MAX_JOBS=126 pip3 install -e python --verbose --no-build-isolation --use-pep517
    ```
    (Note: You have to first build non-AOT version before building AOT version, once you build AOT version, you will always build for AOT in future. To unset this, you have to remove your build directory: `python/build`)


### Test NVIDIA Installation

#### Quick Validation Tests
```sh
# Basic distributed wait test
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma

# NVSHMEM API test
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_nvshmem_api.py
```

#### AllGather GEMM Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case check
bash ./scripts/launch.sh --nproc_per_node 2 python/triton_dist/test/nvidia/test_ag_gemm.py --case check
```

#### GEMM ReduceScatter Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_gemm_rs.py -M 8192 -N 8192 -K 29568 --check
```

#### AllReduce Tests
```sh
NVSHMEM_DISABLE_CUDA_VMM=1 bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_allreduce.py --method one_shot --stress --iters 2
```

#### Flash Decoding Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_decode_attn.py --case perf_8k
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_decode_attn.py --case correctness
```

#### MoE Tests
```sh
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --iters 10 --warmup_iters 20
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2
```

#### E2E Tests
```sh
# Dense model
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_tp_e2e.py --bsz 8 --seq_len 256 --model <model_path> --check --mode ag_rs

# E2E inference
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_e2e_inference.py --bsz 4096 --gen_len 128 --max_length 150 --model <model_path> --backend triton_dist
```

### Run All Unit Tests
The full test suite is available via:
```sh
bash .codebase/scripts/nvidia/run_unittest.sh
```

### Run E2E Tests
```sh
bash .codebase/scripts/nvidia/run_e2e_test.sh
```

### Run Tutorial Tests
```sh
bash .codebase/scripts/nvidia/run_tutorial_test.sh
```

### Run All The Tutorials
See examples in the `tutorials` directory at the project root.

## To use Triton-distributed with the AMD backend:
Starting from the rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 Docker container
#### AMD Build Steps
1. Clone the repo
```sh
git clone https://github.com/ByteDance-Seed/Triton-distributed.git
```
2. Update submodules
```sh
cd Triton-distributed/
git submodule update --init --recursive
```
3. Install dependencies
```sh
sudo apt-get update -y
sudo apt install -y libopenmpi-dev
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3 --no-deps
bash ./shmem/rocshmem_bind/build.sh
python3 -m pip install -i https://test.pypi.org/simple hip-python>=6.3.0 # (or whatever Rocm version you have)
pip3 install pybind11
```
4. Build Triton-distributed
```sh
pip3 install -e python --verbose --no-build-isolation --use-pep517
```
### Test AMD Installation
#### GEMM ReduceScatter example on single node
```sh
bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
 ```
and see the following (reduced) output
```sh
✅ Triton and Torch match
```
