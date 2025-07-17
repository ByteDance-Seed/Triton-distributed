# AutoTuner of Triton-distributed
## Triton Kernel AutoTuner

对于单个 Triton kernel 的 tuning，用户可以直接使用 Triton 原有的接口 [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html) 。

```python
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def some_kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
    ...
```

`triton.autotune` 会在 kernel 运行时开启 tuning 过程，tuning 时会遍历 `configs` 所指定的 tuning config，对于每个 config，运行若干次该 config 对应的 kernel，测量其运行时间，最后会选择平均运行时间最短的 config 作为最优的 config。最优 config 会被保存下来，用 `key` 所指定的参数索引。

## Contextual/Global AutoTuner
tuning 过程需要反复执行某一段代码，这通常要求该段代码是 side-effect-free 的，不会依赖于或者改变某些上下文状态/全局状态，重复执行多次都能成功执行并且产生相同的结果。`triton.autotune` 作用于单个 Triton kernel，有时候我们并不能保证单个 Triton kernel 是 side-effect-free 的。此外，在分布式场景下，不同 rank 的 tuning 结果需要汇总起来，从而选出一个相同的最优 config。因此，用户可能需要一个更通用的 AutoTuner。

因此，我们为用户提供了 `triton_dist.autotuner.contextual_autotune` 接口（`ContextualAutotuner`），可以装饰任意一个无参数函数 `fn`（一个 [Thunk](https://en.wikipedia.org/wiki/Thunk)），该函数的子过程可能不是 side-effect-free 的，不能单独进行 tuning，但是该函数作为一个整体可以进行 tuning。`ContextualAutotuner` 接收 `is_dist` 参数来指定当前 tuning 是否为分布式场景。

### Example

下面是一个基础的 allgather-gemm 的代码，包含 `kernel_local_copy_and_barrier_all` 和  `kernel_consumer_gemm_persistent` 这两个 Triton kernel：

```python
import os
from typing import Optional, List
from cuda import cuda, cudart
import datetime

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.kernels.nvidia.common_ops import barrier_all
from triton_dist.utils import init_nvshmem_by_torch_process_group

import nvshmem.bindings
import nvshmem.core


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            dst = remote_tensor_buffers[rank][
                src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :
            ]
            src = remote_tensor_buffers[src_rank][
                src_rank * M_per_rank : (src_rank + 1) * M_per_rank, :
            ]
            dst.copy_(src)

            (err,) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


@triton.jit
def kernel_local_copy_and_barrier_all(
    rank,
    num_ranks,
    local_buf_ptr,
    global_buf_ptr,
    barrier_ptr,
    M_per_rank,
    N,
    stride_local_m,
    stride_local_n,
    stride_global_m,
    stride_global_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    sm_id = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = sm_id // num_pid_n
    pid_n = sm_id % num_pid_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    data_ptr = (
        local_buf_ptr
        + (pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_local_m
        + (pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_local_n
    )
    dst_ptr = (
        global_buf_ptr
        + (rank * M_per_rank + pid_m * BLOCK_SIZE_M + offs_m[:, None]) * stride_global_m
        + (pid_n * BLOCK_SIZE_N + offs_n[None, :]) * stride_global_n
    )
    mask_data = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (
        pid_n * BLOCK_SIZE_N + offs_n[None, :] < N
    )
    mask_dst = (pid_m * BLOCK_SIZE_M + offs_m[:, None] < M_per_rank) & (
        pid_n * BLOCK_SIZE_N + offs_n[None, :] < N
    )

    data = tl.load(data_ptr, mask=mask_data)
    tl.store(dst_ptr, data, mask=mask_dst)


def local_copy_and_barrier_all(
    rank, num_ranks, local_data, global_data, comm_buf, barrier_ptr, M_per_rank, N
):
    barrier_all[(1,)](rank, num_ranks, comm_buf)
    grid = lambda META: (
        triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_local_copy_and_barrier_all[grid](
        rank,
        num_ranks,
        local_data,
        global_data,
        barrier_ptr,
        M_per_rank,
        N,
        local_data.stride(0),
        local_data.stride(1),
        global_data.stride(0),
        global_data.stride(1),
        128,
        256,
    )
    barrier_ptr.fill_(0)
    # global_data[rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(local_data)
    (err,) = cuda.cuStreamWriteValue32(
        torch.cuda.current_stream().cuda_stream,
        barrier_ptr[rank].data_ptr(),
        1,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    CUDA_CHECK(err)
    barrier_all[(1,)](rank, num_ranks, comm_buf)


@triton.jit
def kernel_consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr,
    comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            alpha = 0
            beta = 0
            pid_m = (
                pid_m + ((((rank ^ alpha) + beta) % num_ranks) * pid_ms_per_rank)
            ) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            token = dl.wait(
                ready_ptr + rank_beg, rank_end - rank_beg + 1, "gpu", "acquire"
            )
            a_desc = dl.consume_token(a_desc, token)

        # You can also put the barrier here with a minor performance drop
        # if needs_wait:
        #     num_barriers_to_wait = num_barriers_wait_per_block
        #     token = dl.wait(ready_ptr + (ki * BLOCK_SIZE_K) // (K // num_ranks), num_barriers_to_wait, "gpu", "acquire")
        #     a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K

        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def ag_gemm_persistent(
    a,
    b,
    c,
    rank,
    num_ranks,
    workspace_tensors,
    barrier_tensors,
    comm_buf,
    ag_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    stages=3,
):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    cp_engine_producer_all_gather_full_mesh_push(
        rank,
        num_ranks,
        a,
        workspace_tensors,
        ag_stream,
        barrier_tensors,
    )
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
        ),
    )
    kernel_consumer_gemm_persistent[grid](
        workspace_tensors[rank],
        b,
        c,
        M,
        N_per_rank,
        K,
        rank,
        num_ranks,
        barrier_tensors[rank],
        comm_buf,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        8,
        False,
        NUM_SMS=NUM_SMS,
        num_stages=stages,
        num_warps=8,
    )

    current_stream.wait_stream(ag_stream)


def test_ag_gemm_tma_intra_node(rank, num_ranks, default_group):
    device = "cuda"
    dtype = torch.float16
    M = 999 * num_ranks
    N = 1024
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    workspace = nvshmem_create_tensors([M, K], dtype=dtype, rank, num_ranks)
    workspace = workspaces[rank]
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    barriers = nvshmem_create_tensor([num_ranks], dtype=torch.int32, rank, num_ranks)
    barrier = barriers[rank]
    barrier.fill_(0)

    # at most 65536 blocks, each block world_size barriers
    max_blocks = 65536
    comm_buf = nvshmem_create_tensor([max_blocks * num_ranks], dtype=torch.int32)
    comm_buf.fill_(0)
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=current_stream)
    torch.cuda.synchronize()

    ag_stream = torch.cuda.Stream()

    def run_ag_gemm_persistent():
        C = torch.empty([M, N_per_rank], dtype=dtype, device=device)
        # Use our own customized barrier kernel
        local_copy_and_barrier_all(
            rank,
            num_ranks,
            A,
            workspace,
            comm_buf,
            barrier,
            M_per_rank,
            K,
        )
        ag_gemm_persistent(
            A,
            B,
            C,
            rank,
            num_ranks,
            workspaces,
            barriers,
            comm_buf,
            ag_stream=ag_stream,
        )
        return C

    current_stream = torch.cuda.current_stream()

    A.copy_(torch.randn([M_per_rank, K], dtype=dtype, device=device))
    B.copy_(torch.randn([N_per_rank, K], dtype=dtype, device=device))
    workspace.copy_(torch.randn([M, K], dtype=dtype, device=device))
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=current_stream)
    torch.cuda.synchronize()
    C = run_ag_gemm_persistent()

    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    for i in range(num_ranks):
        torch.distributed.barrier(default_group)
        if rank == i:
            print(f"Rank {rank}")
            if not torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3):
                print("Golden")
                print(C_golden)
                print("Output")
                print(C)
                print("Max diff", torch.max(torch.abs(C_golden - C)))
                print("Avg diff", torch.mean(torch.abs(C_golden - C)))
                print("Wrong Answer!")
            else:
                print("Pass!")


def main():
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)), backend="nccl"
    )
    torch.distributed.barrier(TP_GROUP)

    torch.cuda.synchronize()
    init_nvshmem_by_torch_process_group(TP_GROUP)

    test_ag_gemm_tma_intra_node(RANK, WORLD_SIZE, TP_GROUP)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
```

可以用下面的命令来测试上述代码：

```bash
bash ./scripts/launch.sh <file_name>
```

下面我们给 `kernel_consumer_gemm_persistent` 添加 `triton.autotune`，对 `kernel_consumer_gemm_persistent` 进行修改：

```python
def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in [3, 4]
        for w in [4, 8]
    ]


@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])
@triton.jit
def kernel_consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    ready_ptr,
    comm_buf_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    ...

def ag_gemm_persistent(
    a,
    b,
    c,
    rank,
    num_ranks,
    workspace_tensors,
    barrier_tensors,
    comm_buf,
    ag_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    stages=3,
):
    ...
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
        ),
    )
    kernel_consumer_gemm_persistent[grid](
        workspace_tensors[rank],
        b,
        c,
        M,
        N_per_rank,
        K,
        rank,
        num_ranks,
        barrier_tensors[rank],
        comm_buf,
        # BLOCK_M,
        # BLOCK_N,
        # BLOCK_K,
        # 8,
        EPILOGUE_SUBTILE=False,
        NUM_SMS=NUM_SMS,
        # num_stages=stages,
        # num_warps=8,
    )

    current_stream.wait_stream(ag_stream)
```

考虑到 `kernel_consumer_gemm_persistent` 是 `run_ag_gemm_persistent` 的一个子过程，而 `run_ag_gemm_persistent` 需要作为一个整体运行。为此，我们只需要用  `triton_dist.autotuner.contextual_autotune` 装饰 `run_ag_gemm_persistent` 函数（并且设 `is_dist=True`）：

```python

from triton_dist.autotuner import contextual_autotune

def test_ag_gemm_tma_intra_node(rank, num_ranks, default_group):
    ...

    @contextual_autotune(is_dist=True)
    def run_ag_gemm_persistent():
        ...

    ...
```

其中，rank-i 的 tuning 过程的 log 输出会打印在 `./.autotune_logs/rank-i.log` 中。

更多的例子可以参考部分测试文件：[test_ag_gemm.py](../../python/triton_dist/test/nvidia/test_ag_gemm.py)、[test_moe_reduce_rs.py](../../python/triton_dist/test/nvidia/test_moe_reduce_rs.py)、[test_ag_moe.py](../../python/triton_dist/test/nvidia/test_ag_moe.py)，可以用如下命令进行测试：

```bash
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_gemm.py --case correctness_tma_autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_moe_reduce_rs.py 8192 2048 1536 32 2 --check --autotune
bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_ag_moe.py --M 2048 --autotune
```

### Implementaiton

`ContextualAutotuner` 会检测 `fn` 运行过程中被 `triton.autotune` 所装饰的并且触发 tuning 的 Triton kernel 并为其构建 tuning-context 来维护其当前的 tuning 状态。`ContextualAutotuner` 会多次运行 `fn`，每次运行过程中每个 Triton kernel 的一次调用都只会触发一次 kernel launch，该 kernel 对应的 tuning-context 会记录这次 launch 的测量时间以及 tuning 的进度，在该 kernel 的每个 config 都被充分测量之后，该 kernel 根据记录的测量时间来选出最优的 config 并保存。

下面展示一个例子，假设 `fn` 每次运行过程中会各调用一次 kernel-0 和 kernel-1，kernel-0 一共有两个 tuning-config，kernel-1 有三个 tuning-config，每个 config 的测量次数为 2 次（进行两次 tuning-iter），则 `ContextualAutotuner` 的 tuning 过程就如下面表格所示：

| Tuning-Iter                    |     |                                   |     |                                   |     |
| ------------------------------ | --- | --------------------------------- | --- | --------------------------------- | --- |
| 0                              | ... | kernel-0 (config-0 (iter-0))      | ... | kernel-1 (config-0 (iter-0))      | ... |
| 1                              | ... | kernel-0 (config-0 (iter-1))      | ... | kernel-1 (config-0 (iter-1))      | ... |
| 2                              | ... | kernel-0 (config-1 (iter-0))      | ... | kernel-1 (config-1 (iter-0))      | ... |
| 3                              | ... | kernel-0 (config-1 (iter-1))      | ... | kernel-1 (config-1 (iter-1))      | ... |
| 4                              | ... | kernel-0 (config-0 (best-config)) | ... | kernel-1 (config-2 (iter-0))      | ... |
| 5                              | ... | kernel-0 (config-0 (best-config)) | ... | kernel-1 (config-2 (iter-1))      | ... |
| final execution to get results | ... | kernel-0 (config-0 (best-config)) | ... | kernel-1 (config-1 (best-config)) | ... |

注意，在 `ContextualAutotuner` 执行 Tuning-Iter-3 的时候，kernel-0 的所有 config 都已经测量完毕，选出了 config-0 作为 best-config，而 kernel-1 的 config 还没测量完毕，所以 `ContextualAutotuner` 会继续执行 `fn` 直到 kernel-1 也 tuning 结束。
