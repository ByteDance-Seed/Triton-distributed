# Build Triton-distributed

## The best practice to use Triton-distributed with the Nvidia backend:
- Python >=3.11 (suggest using virtual environment)
- CUDA >=12.4
- Torch >=2.4.1
- Clang >=19

We recommend installation in [Nvidia PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).

#### if for AMD GPU:
- ROCM 6.3.0
- Torch 2.4.1 with ROCM support



Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

### Steps:
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
    pip3 install torch==2.4.1
    pip3 install cuda-python==12.4 # need to align with your nvcc version
    pip3 install ninja cmake wheel pybind11 numpy chardet pytest
    pip3 install pynvml>=11.5.3
    ```
5. Install NVSHMEM.
    
    ```
    pip3 install nvidia-nvshmem-cu12==3.3.9 cuda.core==0.2.0 "Cython>=0.29.24"

    CPPFLAGS="-I/usr/local/cuda/include" pip3 install https://developer.download.nvidia.com/compute/nvshmem/redist/nvshmem_python/source/nvshmem_python-source-0.1.0.36132199_cuda12-archive.tar.xz
    ```

6. Build Triton-distributed

    Then you can build Triton-distributed.
    ```sh
    # Remove triton installed with torch
    pip uninstall triton
    pip uninstall triton_dist # remove previous triton-dist
    rm -rf /usr/local/lib/python3.12/dist-packages/triton
    # Install Triton-distributed
    cd /workspace/Triton-distributed
    export USE_TRITON_DISTRIBUTED_AOT=0
    pip3 install -e python --verbose --no-build-isolation --use-pep517
    ```

    We also provide AOT version of Triton-distributed. If you want to use AOT (**Not Recommended**), then
    ```sh
    cd /workspace/Triton-distributed/
    source scripts/setenv.sh
    bash scripts/gen_aot_code.sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    pip3 install -e python --verbose --no-build-isolation --use-pep517
    ```
    (Note: You have to first build non-AOT version before building AOT version, once you build AOT version, you will always build for AOT in future. To unset this, you have to remove your build directory: `python/build`)

7. Setup environment variables (Optional)
    ```sh
    cd /home/Triton-distributed
    source scripts/setenv.sh
    ```

### Test your installation
#### AllGather GEMM example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_distributed_wait.py --case correctness_tma
```

#### GEMM ReduceScatter example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_gemm_rs.py 8192 8192 29568
```

#### NVSHMEM example in Triton-distributed
```sh
bash ./scripts/launch.sh ./python/triton_dist/test/nvidia/test_nvshmem_api.py
```

### Run All The Tutorials
See examples in [`tutorials`](../tutorials/README.md)

## To use Triton-distributed with the AMD backend:
Starting from the rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 Docker container
#### Steps:
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
### Test your installation
#### GEMM ReduceScatter example on single node
```sh
bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ag_gemm_intra_node.py 8192 8192 29568
 ```
and see the following (reduced) output
```sh
✅ Triton and Torch match
```