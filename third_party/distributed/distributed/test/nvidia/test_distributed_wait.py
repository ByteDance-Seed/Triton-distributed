################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import torch
import triton
import triton.language as tl
import triton.distributed.language as dl

import time
import argparse
import os
import sys
from typing import Optional

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


@triton.jit
def kernel_consumer_gemm(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Distributed parameters
    rank,
    num_ranks,
    ready_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    K_per_barrier,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    is_fp8: tl.constexpr,
    needs_wait: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    dtype = tl.float16 if not is_fp8 else tl.float8e4nv
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    if needs_wait:
        num_barriers_to_wait = 1
        token = dl.wait(ready_ptr, num_barriers_to_wait, "gpu", "acquire")
        a_ptrs = dl.consume_token(a_ptrs, token)
    for k in range(0, num_k_blocks):
        # You can also put barriers here (but with performance drawback)
        # if needs_wait:
        #     num_barriers_to_wait = 1
        #     token = dl.wait(ready_ptr + ((k * BLOCK_SIZE_K) // K_per_barrier), num_barriers_to_wait, "gpu", "acquire")
        #     a_ptrs = dl.consume_token(a_ptrs, token)
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(dtype)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def consumer_gemm(A, B, C, rank, num_ranks, barrier, needs_wait=True):
    M, K = A.shape
    _, N = B.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    assert K % num_ranks == 0
    compiled = kernel_consumer_gemm[grid](A, B, C, rank, num_ranks, barrier, M, N, K, K // num_ranks, A.stride(0),
                                          A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1), 128, 128, 32,
                                          8, False, needs_wait, num_stages=4)
    return compiled


@register_test("lower")
def test_lower_wait(args):
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"

    device = "cuda"
    dtype = torch.float16

    rank = 0
    num_ranks = 8
    barrier_tensor = torch.ones([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 1024

    assert M % num_ranks == 0
    N_per_rank = N // num_ranks

    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([K, N_per_rank], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    compiled = consumer_gemm(ag_A, B, C, rank, num_ranks, barrier_tensor)
    print(compiled.asm["ptx"])

    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MLIR_ENABLE_DUMP"] = "0"


@register_test("correctness")
def test_1024_gemm_single_device(args):
    device = "cuda"
    dtype = torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barrier_tensor = torch.zeros([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 256

    assert M % num_ranks == 0
    N_per_rank = N // num_ranks

    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([K, N_per_rank], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    C_golden = torch.matmul(ag_A, B)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        consumer_gemm(ag_A, B, C, rank, num_ranks, barrier_tensor)

    print("Consumer GEMM launched!")
    print("signals are:")
    print(barrier_tensor)
    print("sleeping...", flush=True)
    time.sleep(3)
    print("wake up!", flush=True)
    barrier_tensor.fill_(1)
    print("signals are:")
    print(barrier_tensor)

    torch.cuda.current_stream().wait_stream(stream)
    print(barrier_tensor)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")


def measure_cuda_function_performance(cuda_function, warmup=10, repeat=100):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    for _ in range(warmup):
        cuda_function()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(repeat):
        start_event.record()
        cuda_function()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        total_time += elapsed_time

    avg_time = total_time / repeat
    return avg_time


@register_test("perf")
def test_perf_gemm_single_device(args):
    device = "cuda"
    dtype = torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barrier_tensor = torch.zeros([num_ranks], dtype=torch.int32, device=device)
    barrier_tensor.fill_(1)
    shapes = [256 * 2**i for i in range(7)]
    perfs = []

    for needs_wait in [True, False]:
        print(f"With Barrier {needs_wait}\nShape Perf(ms)", flush=True)
        for shape in shapes:
            M = shape
            N = shape
            K = shape

            A = torch.randn([M, K], dtype=dtype, device=device)
            B = torch.randn([K, N], dtype=dtype, device=device)
            C = torch.empty([M, N], dtype=dtype, device=device)

            C_golden = torch.matmul(A, B)

            consumer_gemm(A, B, C, rank, num_ranks, barrier_tensor, needs_wait)
            assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3), f"Correctness not passed for shape {shape}"

            avg_time = measure_cuda_function_performance(
                lambda *x: consumer_gemm(A, B, C, rank, num_ranks, barrier_tensor, needs_wait), warmup=10, repeat=100)
            perfs.append(avg_time)
            print(shape, avg_time, flush=True)
        print()


# TMA related test
def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    rank,
    num_ranks,
    ready_ptr,
    num_barriers_wait_per_block,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
    needs_wait: tl.constexpr,
):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
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

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            if needs_wait:
                num_barriers_to_wait = num_barriers_wait_per_block
                token = dl.wait(ready_ptr, num_barriers_to_wait, "gpu", "acquire")
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


def consumer_gemm_persistent(a, b, c, rank, num_ranks, barrier_tensor, needs_wait=True, barriers_per_block=1):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    compiled = kernel_consumer_gemm_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        rank,
        num_ranks,
        barrier_tensor,
        barriers_per_block,
        128,
        256,
        64,
        8,
        False,
        NUM_SMS=NUM_SMS,  #
        needs_wait=needs_wait,
        num_stages=3,
    )
    return compiled


@register_test("lower_tma")
def test_lower_tma_wait(args):
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"

    device = "cuda"
    dtype = torch.float16

    rank = 0
    num_ranks = 8
    barrier_tensor = torch.ones([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 1024

    assert M % num_ranks == 0
    N_per_rank = N // num_ranks

    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    compiled = consumer_gemm_persistent(ag_A, B, C, rank, num_ranks, barrier_tensor)
    print(compiled.asm["ptx"])

    os.environ["TRITON_ALWAYS_COMPILE"] = "0"
    os.environ["MLIR_ENABLE_DUMP"] = "0"


@register_test("correctness_tma")
def test_1024_gemm_tma_single_device(args):
    device = "cuda"
    dtype = torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barrier_tensor = torch.zeros([num_ranks], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 256

    assert M % num_ranks == 0
    N_per_rank = N // num_ranks

    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    C_golden = torch.matmul(ag_A, B.T)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        consumer_gemm_persistent(ag_A, B, C, rank, num_ranks, barrier_tensor)

    print("Consumer GEMM launched!")
    print("signals are:")
    print(barrier_tensor)
    print("sleeping...", flush=True)
    time.sleep(3)
    print("wake up!", flush=True)
    barrier_tensor.fill_(1)
    print("signals are:")
    print(barrier_tensor)

    torch.cuda.current_stream().wait_stream(stream)
    print(barrier_tensor)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")


@register_test("correctness_tma_multi_barrier")
def test_1024_gemm_tma_single_device_multi_barrier(args):
    device = "cuda"
    dtype = torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barriers_per_block = 4
    barrier_tensor = torch.zeros([num_ranks + barriers_per_block - 1], dtype=torch.int32, device=device)
    M = 1024
    N = 1024
    K = 256

    assert M % num_ranks == 0
    N_per_rank = N // num_ranks

    ag_A = torch.randn([M, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)
    C = torch.empty([M, N_per_rank], dtype=dtype, device=device)

    C_golden = torch.matmul(ag_A, B.T)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        consumer_gemm_persistent(ag_A, B, C, rank, num_ranks, barrier_tensor, needs_wait=True,
                                 barriers_per_block=barriers_per_block)

    print("Consumer GEMM launched!")
    print("signals are:")
    print(barrier_tensor)
    print("sleeping...", flush=True)
    time.sleep(3)
    print("wake up!", flush=True)
    barrier_tensor.fill_(1)
    print("signals are:")
    print(barrier_tensor)

    torch.cuda.current_stream().wait_stream(stream)
    print(barrier_tensor)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")


@register_test("perf_tma")
def test_perf_gemm_tma_single_device(args):
    device = "cuda"
    dtype = torch.float16
    rank = 0
    num_ranks = 8
    # TODO(zhengsize): why needs + 1?
    barrier_tensor = torch.zeros([num_ranks], dtype=torch.int32, device=device)
    barrier_tensor.fill_(1)
    shapes = [256 * 2**i for i in range(7)]
    perfs = []

    for needs_wait in [True, False]:
        print(f"With Barrier {needs_wait}\nShape Perf(ms)", flush=True)
        for shape in shapes:
            M = shape
            N = shape
            K = shape

            A = torch.randn([M, K], dtype=dtype, device=device)
            B = torch.randn([N, K], dtype=dtype, device=device)
            C = torch.empty([M, N], dtype=dtype, device=device)

            C_golden = torch.matmul(A, B.T)

            consumer_gemm_persistent(A, B, C, rank, num_ranks, barrier_tensor, needs_wait)
            assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3), f"Correctness not passed for shape {shape}"

            avg_time = measure_cuda_function_performance(
                lambda *X: consumer_gemm_persistent(A, B, C, rank, num_ranks, barrier_tensor, needs_wait), warmup=10,
                repeat=100)
            perfs.append(avg_time)
            print(shape, avg_time, flush=True)
        print()


if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)
