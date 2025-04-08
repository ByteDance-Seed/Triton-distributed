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

[Build Guide](docs/distributed/build.md)

### How to use Triton-distributed
Triton-distributed provides a set of easy-to use primitives to support the development of distributed compute-communication overlapping kernels. The primitives are divided into low-level primitives and high-level primitives. Currently, we have released our low-level primitives, and we plan to release high-level primitives in future.

[Triton-distributed Primitives](docs/distributed/primitives.md)

Using these primitives, users can program communication kernels easily. For example, a low-latency AllToAll (with better latency than [DeepEP](https://github.com/deepseek-ai/DeepEP) for inference) is shown below.
The performance of this example on 32 H800 GPUs is 137us (128 tokens per rank, topk=8, hidden_size=7168, dtype=fp8), while DeepEP is 182 us (note: DeepEP doesn't use NVLink for inference).
```py
@triton.jit
def all_to_all_kernel(
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    splits_cumsum,
    scale_src,
    scale_dst,
    rank: int,
    call_count: int,
    WITH_SCALE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
    SCALE_ELEMENT_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    threadidx = tid(axis=0)

    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK

    m_st = tl.load(splits_cumsum + exp_st)
    m_ed = tl.load(splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    src_off = m_st
    dst_off = rank * MAX_M

    split_src_ptr = splits_src + exp_st
    off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
    off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
    cumsum_sts = tl.load(splits_cumsum + off0)
    cumsum_eds = tl.load(splits_cumsum + off1)
    tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)

    act_pos = call_count % 2
    data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
    split_dst_ptr = splits_dst + act_pos * NUM_TOT_EXPERTS + rank * EXPERTS_PER_RANK
    signal_ptr = signal + act_pos * WORLD_SIZE + rank

    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src + src_off * HIDDEN,
        num_rows_cur_block * HIDDEN * ELEMENT_SIZE,
        pid,
    )
    libshmem_device.putmem_nbi_block(
        split_dst_ptr,
        split_src_ptr,
        EXPERTS_PER_RANK * 4,  # now we use `int32` for splits
        pid,
    )
    if WITH_SCALE:
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M + dst_off
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src + src_off,
            num_rows_cur_block * SCALE_ELEMENT_SIZE,
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    libshmem_device.fence()
    if threadidx == 0:
        if not WITH_SCALE:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )
```

Also, users can combine the communication part with computation part to design overlapping kernels. We have provided example implementations in `third_party/distributed/distributed/kernels`.

## Performance
Triton-distributed can achieve comparable or better performance than hand-tuned libraries.


### AllGather GEMM on single node of H800x8
![Ag-GEMM-inter-node](asset/ag-gemm-intra-node.png)

### GEMM ReduceScatter on single node of H800x8
![Ag-GEMM-inter-node](asset/gemm-rs-intranode-perf.png)

### AllGather GEMM on 2 nodes of H800x8
![Ag-GEMM-inter-node](asset/ag-inter-node-gemm.png)

### GEMM ReduceScatter on 2 nodes of H800x8
![GEMM-Rs-inter-node](asset/gemm-rs-inter-node.png)

### Scaling of Distributed Flash-Decode from 1 GPU to 32 GPUs
The batch size is 1 (one query) for decoding.
![flash-decode-inter-node](asset/flash-decode-scaling.png)

### Performance on Other Platforms
[AMD GPUs](docs/distributed/amd-perf.md)


## Roadmaps
### Functionalities
- [x] Release low-level primitives
- [ ] Release high-level primitives
- [ ] Tutorials
- [ ] Pre-built binary
### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [x] Release single-node MoE TP overlapping kernels
- [x] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [x] Release cross-node GEMM TP overlapping kernels
- [x] Release cross-node MoE TP overlapping kernels
- [x] Release cross-node distributed Flash-Decoding kernels
- [x] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
- [ ] Provide tutorials for kernel implementation
### Backends
Computation
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [x] Nvidia SM89 support
- [x] AMD CDNA3 support

Communication
- [x] NVLink
- [x] IB
- [ ] PCIe 
### Performance
- [ ] Performance report

## License
The Triton-distributed project is under MIT license.
Part of our code is under Apache-2.0 License:
- `third_party/distributed/distributed/kernels/flash_decode.py`

Triton's original code is partially under Apache-2.0 License, these files include:
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

# Discussion and Contribution
Please use issues or pull requests for discussion and contribution (see [CONTRIBUTING.md](CONTRIBUTING.md)).
