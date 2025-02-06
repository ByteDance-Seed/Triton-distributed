<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# Triton-distributed

<!-- <p align="center">
  <a href="https://github.com/bytedance/flux">
    <img src="https://img.shields.io/badge/Triton-distributed-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/xxxx.xxxx">
    <img src="https://img.shields.io/badge/Triton-distributed-Tech Report-red"></a>
  <br>
  <a href="https://github.com/user-attachments/assets/d3fcb3bf-466b-4efe-8c3f-5f85258202ae">
    <img src="https://img.shields.io/badge/Triton-distributed-Wechat Communication Group-07C160"></a>
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-MIT-blue"></a>
</p> -->

[Original Triton README](upstream-README.md) | [README in Chinese](README-cn.md)

Triton-distributed is a distributed compiler designed for computation-communication overlapping, which is based on OpenAI Triton.

Using Triton-distributed, programmers are able to develop efficient kernels comparable to highly-optimized libraries (including [Distributed-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm) and [FLUX](https://github.com/bytedance/flux/blob/main/README.md)).
Triton-distributed currently mainly targets Nvidia GPU and AMD GPU. It can also be ported to other hardware platforms.
Feel free to contact us if you want to use Triton-distributed on your own hardware.

## Getting started
### Install Triton-distributed from source
#### The best practice to use Triton-distributed:
- Python 3.9 (suggest using virtual environment)
- CUDA 12.4
- Torch 2.4

Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

#### Steps:
1. Clone Triton-distributed to your own path (e.g., `/home/Triton-distributed`)
2. Update submodules
    ```sh
    git submodule update --init --recursive
    ```
3. Install dependencies
    ```sh
    pip3 install torch==2.4
    pip3 install black "clang-format==19.1.2" pre-commit ruff yapf==0.43
    pip3 install ninja cmake wheel pybind11 cuda-python==12.4 numpy chardet pytest
    ```
4. Apply NVSHMEM fix
(Disclaimer: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.)

    1. Download NVSHMEM 3.2.5 Source Code [NVSHMEM Open Source Packages](https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz)
    2. Extract to designated location
        ```sh
        mkdir -p /home/Triton-distributed/third_party/nvshmem
        tar -xvf nvshmem_src_3.2.5-1.txz -C /home/Triton-distributed/third_party/nvshmem/ --strip-components=1
        ```
    3. Bitcode Bug Fix: [BUG with nvshmem 3.2.5 for bitcode compiling](https://forums.developer.nvidia.com/t/bug-with-nvshmem-3-2-5-for-bitcode-compiling/327847)

       File: ```src/include/non_abi/device/common/nvshmemi_common_device.cuh``` (Line 287)
       ```cpp
        - dst = (void *)(dst_p + nelems);
        - src = (void *)(src_p + nelems);

        +#ifdef __clang_llvm_bitcode_lib__
        +    dst = (void *)(dst_p + nelems * 4);
        +    src = (void *)(src_p + nelems * 4);
        +#else
        +    dst = (void *)(dst_p + nelems);
        +    src = (void *)(src_p + nelems);
        +#endif
        ```
    4. Clang Compilation Error Fix

       File: ```src/include/device_host/nvshmem_common.cuh``` (Line 41)
       ```cpp
        - __device__ int __nvvm_reflect(const char *s);
        + __device__ int __nvvm_reflect(const void *s);
       ```

5. Build
    ```sh
    cd /home/Triton-distributed
    export USE_TRITON_DISTRIBUTED_AOT=0
    pip3 install -e python --verbose --no-build-isolation
    ```

    If you want to use AOT, then
    ```sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    pip3 install -e python --verbose --no-build-isolation
    ```
    (Note: You have to first build non-AOT version before building AOT version)
6. Setup environment variables (Do this step at the beginning every time you use Triton-distributed)
    ```sh
    cd /home/Triton-distributed
    source scripts/setenv.sh
    ```

### Test your installation
#### AllGather GEMM example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/nvidia/test_ag_gemm_intra_node.py --case correctness_tma
```
#### GEMM ReduceScatter example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/nvidia/test_gemm_rs_multi_node.py 8192 8192 29568
```
#### NVSHMEM example in Triton-distributed
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/nvidia/test_nvshmem_api.py
```

### How to use Triton-distributed
Triton-distributed provides a set of easy-to use primitives to support the development of distributed compute-communication overlapping kernels. The primitives are divided into low-level primitives and high-level primitives. Currently, we have released our low-level primitives, and we plan to release high-level primitives in future.

All the primitives are exposed by `triton.distributed.language`
#### Low-level primitives
##### Context Querying Primitives
```py
rank(axis=-1, _builder=None)
num_ranks(axis=-1, _builder=None)
symm_at(ptr, rank, _builder=None)

```
##### Singal Control Primitives
```py
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)
consume_token(value, token, _builder=None)
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```
##### NVSHMEM-related Primitives

Besides the primitives, Triton-distributed also expose all the NVSHMEM primitives to Python, allowing users to program communication kernels purely in Python.

All the NVSHMEM-related device-side primitives are exposed by `triton.language.extra.libshmem_device`
```py
my_pe()
n_pes()
int_p(dest, value, pe)
remote_ptr(local_ptr, pe)
barrier_all()
barrier_all_block()
barrier_all_warp()
sync_all()
sync_all_block()
sync_all_warp()
quiet()
fence()
getmem_nbi_block(dest, source, bytes, pe)
getmem_block(dest, source, bytes, pe)
getmem_nbi_warp(dest, source, bytes, pe)
getmem_warp(dest, source, bytes, pe)
getmem_nbi(dest, source, bytes, pe)
getmem(dest, source, bytes, pe)
putmem_block(dest, source, bytes, pe)
putmem_nbi_block(dest, source, bytes, pe)
putmem_warp(dest, source, bytes, pe)
putmem_nbi_warp(dest, source, bytes, pe)
putmem(dest, source, bytes, pe)
putmem_nbi(dest, source, bytes, pe)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
signal_op(sig_addr, signal, sig_op, pe)
signal_wait_until(sig_addr, cmp_, cmp_val)
```

Using these primitives, users can program compute-communication kernels easily. For example, a ring-put example is shown here:
```py
@triton.jit
def ring_put(ptr):
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)
```

#### High-level primitives
To provide better programming experience, we also provide a set of high-level primitives for communication and signal control. These primitives, as decribed in our [MLSys 2025 paper](https://mlsys.org/virtual/2025/poster/2969), use a tile-centric design philosophy. These high-level primitives will be released soon after MLSys 2025.

## Roadmaps
### Functionalities
- [x] Release low-level primitives
- [ ] Release high-level primitives
### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [x] Release single-node MoE TP overlapping kernels
- [x] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [x] Release cross-node GEMM TP overlapping kernels
- [x] Release cross-node MoE TP overlapping kernels
- [x] Release cross-node distributed Flash-Decoding kernels
- [x] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
### Backends
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [ ] Nvidia SM89 support
- [x] AMD CDNA3 support
### Performance
- [ ] Performance report

## License
The Triton-distributed project is under MIT license.
Part of our code is under Apache-2.0 License:
- `third_party/distributed/distributed/kernels/flash_decode.py`
Triton's original code is partially under Apache-2.0 Linces, these files include:
- `include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp`
- `python/triton/_C/include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h`
- `utils/generate-test-checks.py`


## Citation
If you use Triton-distributed in a scientific publication, we encourage you to add the following reference to the related papers:
```bibtex
@misc{zheng2025tilelink,
      title={TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives},
      author={Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Xin Liu},
      year={2025},
}
```

# About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

# Join the Discussion Group

![discussion-group](asset/wechat-group-temporal.png)
