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

# ${SCRIPT_DIR}/../python/triton_dist/test/nvidia/test_compile_aot.py:test_aot_kernel
# ${SCRIPT_DIR}/../python/triton_dist/test/nvidia/test_compile_aot.py:matmul_kernel_descriptor_persistent

from typing import Optional
import os
import warnings
import torch
import triton
import triton.language as tl
from triton_dist.utils import has_tma
from triton_dist.tools.compile_aot import aot_compile_spaces
from triton_dist.utils import initialize_distributed
import nvshmem.core
from triton_dist.kernels.nvidia.common_ops import BarrierAllContext

USE_AOT_ENV = os.environ.get("USE_TRITON_DISTRIBUTED_AOT", "0")
USE_AOT = USE_AOT_ENV.lower() in ["1", "true", "on"]

cdiv = triton.cdiv

NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count


def get_test_aot_kernel_info():
    return {
        "num_warps": 4,
        "num_stages": 4,
    }


@aot_compile_spaces({
    "test_aot_kernel": {
        "signature": "*bf16, *bf16",
        "grid": ["1", "1", "1"],
        "triton_algo_infos": [get_test_aot_kernel_info()],
    }
})
@triton.jit
def test_aot_kernel(
    A,
    B,
):
    x = tl.load(A + tl.arange(0, 64))
    tl.store(B + tl.arange(0, 64), x)


#######################################################


def matmul_tma_persistent_get_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BM": BM,
                "BN": BN,
                "BK": BK,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        ) for BM in [128] for BN in [128, 256] for BK in [64] for s in ([4]) for w in [4]
    ]


def _get_default_num_stages():
    return 4 if torch.cuda.get_device_capability()[0] >= 9 else 2


def get_matmul_kernel_descriptor_persistent_info():
    return {
        "NUM_SMS": NUM_SMS,
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "num_warps": 4,
        "num_stages": _get_default_num_stages(),
    }


@aot_compile_spaces({
    f"matmul_kernel_descriptor_persistent_{ty}": {
        "signature": f"*{ty}, *{ty}, *{ty}, i32, i32, i32, %BM, %BN, %BK, %NUM_SMS",
        "grid": ["%NUM_SMS", "1", "1"],
        "triton_algo_infos": [
            get_matmul_kernel_descriptor_persistent_info(),
        ],
    }
    for ty in ["bf16", "fp16"]
})
@triton.autotune(
    configs=matmul_tma_persistent_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit()
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BM: tl.constexpr = 128,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 64,
    NUM_SMS: tl.constexpr = NUM_SMS,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = tile_id // num_pid_n, tile_id % num_pid_n
        offs_am = pid_m * BM
        offs_bn = pid_n * BN
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BK
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        pid_m, pid_n = tile_id_c // num_pid_n, tile_id_c % num_pid_n
        offs_cm = pid_m * BM
        offs_cn = pid_n * BN
        c = accumulator.to(dtype)
        c_desc.store([offs_cm, offs_cn], c)


#######################################################


def test_test_aot_kernel():
    A = torch.randn(128, dtype=torch.bfloat16)
    B = torch.zeros(128, dtype=torch.bfloat16)
    C = torch.zeros(128, dtype=torch.bfloat16)

    print("🚀️️ TEST_AOT_KERNEL JIT")
    test_aot_kernel[(1, )](
        A=A,
        B=B,
    )
    torch.cuda.synchronize()
    print("🚀️ TEST_AOT_KERNEL AOT")
    algo_info = distributed.test_aot_kernel__triton_algo_info_t()
    for _k, _v in get_test_aot_kernel_info().items():
        setattr(algo_info, _k, _v)
    distributed.test_aot_kernel(
        torch.cuda.current_stream().cuda_stream,
        A.data_ptr(),
        C.data_ptr(),
        algo_info,
    )
    torch.testing.assert_close(B, C, rtol=1e-5, atol=1e-5)
    print("✅ TEST_TEST_AOT_KERNEL done")


def test_barrier_all_intra_node_atomic_cas_block_i32():
    barrier_ctx = BarrierAllContext(is_intra_node=True)
    print("🚀 TEST_BARRIER_ALL_INTRA_NODE_ATOMIC_CAS_BLOCK_I32 AOT")
    algo_info = distributed.barrier_all_intra_node_atomic_cas_block_i32__triton_algo_info_t()
    kv_algo_info = {"num_warps": 4, "num_stages": 2}
    for _k, _v in kv_algo_info.items():
        setattr(algo_info, _k, _v)
    distributed.barrier_all_intra_node_atomic_cas_block_i32(
        0,  # torch.cuda.current_stream().cuda_stream,
        barrier_ctx.local_rank,
        barrier_ctx.rank,
        barrier_ctx.num_local_ranks,
        barrier_ctx.symm_barrier.data_ptr(),
        algo_info,
    )

    barrier_ctx.finalize()
    print("✅ TEST_BARRIER_ALL_INTRA_NODE_ATOMIC_CAS_BLOCK_I32 done")


def test_matmul_descriptor_persistent():
    """
    TMA descriptors require a global memory allocation
    """
    if not has_tma():
        warnings.warn("TMA is not enabled, skip the test")
        return
    dtype = torch.bfloat16
    M, N, K = 2048, 4096, 2048
    a = torch.randn((M, K), dtype=dtype)
    b = torch.randn((N, K), dtype=dtype)
    c = torch.empty((M, N), dtype=dtype)
    d = torch.empty((M, N), dtype=dtype)
    assert cdiv(M, 128) * cdiv(N, 256) > NUM_SMS

    print("🚀️ TEST_MATMUL_DESCRIPTOR_PERSISTENT JIT")

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    matmul_kernel_descriptor_persistent[(NUM_SMS, )](
        a,
        b,
        c,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
    )
    print("🚀 TEST_MATMUL_DESCRIPTOR_PERSISTENT AOT")
    algo_info = distributed.matmul_kernel_descriptor_persistent_bf16__triton_algo_info_t()
    for _k, _v in get_matmul_kernel_descriptor_persistent_info().items():
        setattr(algo_info, _k, _v)
    distributed.matmul_kernel_descriptor_persistent_bf16(
        0,  # torch.cuda.current_stream().cuda_stream,
        a.data_ptr(),
        b.data_ptr(),
        d.data_ptr(),
        M,
        N,
        K,
        algo_info,
    )

    expected = a @ b.T

    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(d, expected, rtol=1e-5, atol=1e-5)
    print("✅ TEST_MATMUL_DESCRIPTOR_PERSISTENT done")


if __name__ == "__main__":
    if USE_AOT:
        try:
            from triton._C.libtriton_distributed import distributed
        except ImportError as e:
            print("AOT lib not found, please follow the doc to build")
            print(e)
    else:
        print("Triton Distributed AOT is not enabled. skip the test")
        exit(0)

    GROUP = initialize_distributed()

    torch.set_default_device("cuda")
    torch.manual_seed(42)

    test_test_aot_kernel()
    test_matmul_descriptor_persistent()
    test_barrier_all_intra_node_atomic_cas_block_i32()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
