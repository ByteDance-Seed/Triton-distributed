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
from triton.tools.tensor_descriptor import TensorDescriptor

from typing import Optional
import triton_dist
import triton_dist.tune
from triton_dist.tune import to_hashable
from triton_dist.utils import get_device_max_shared_memory_size
from triton_dist.kernels.nvidia.gemm_perf_model import get_tensorcore_tflops, get_dram_gbps


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_size_m in [16, 32]:
            for block_size_k in [32, 64]:
                for block_size_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_size_n <= 64 else 4
                    configs.append((block_size_m, block_size_n, block_size_k, 1, num_stages, num_warps))
    return configs


def get_config_space_ada():
    # by tuning GEMM for Llama 7B / Llama 3.1 8B/70B/405B / GPT3 / Qwen
    # TODO(houqi.1993) only for BF16/FP16
    return [
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=6),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
    ]


def get_config_space_a100():
    # by tuning GEMM for Llama 7B / Llama 3.1 8B/70B/405B / GPT3 / Qwen
    # TODO(houqi.1993) only for BF16/FP16
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=8,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=4),
    ]


def get_config_space_default():
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps
    configs = [
        # basic configs for compute-bound matmuls
        (128, 128, 32, 4, 4, 4),
        (128, 128, 32, 8, 4, 4),
        (128, 256, 32, 4, 3, 8),
        (128, 256, 64, 8, 3, 8),
        (128, 32, 32, 4, 4, 4),
        (128, 32, 32, 8, 4, 4),
        (128, 64, 32, 4, 4, 4),
        (128, 64, 32, 8, 4, 4),
        (256, 128, 32, 4, 3, 8),
        (256, 64, 32, 4, 4, 4),
        (32, 64, 32, 8, 5, 2),
        (64, 128, 32, 4, 4, 4),
        (64, 128, 32, 8, 4, 4),
        (64, 256, 32, 4, 4, 4),
        (64, 256, 32, 8, 4, 4),
        (64, 32, 32, 4, 5, 2),
        (64, 32, 32, 8, 5, 2),
        # good for int8/fp8 inputs
        (128, 128, 128, 4, 4, 4),
        (128, 128, 128, 8, 4, 4),
        (128, 256, 128, 4, 3, 8),
        (128, 256, 128, 8, 3, 8),
        (128, 32, 64, 4, 4, 4),
        (128, 32, 64, 8, 4, 4),
        (128, 64, 64, 4, 4, 4),
        (128, 64, 64, 8, 4, 4),
        (256, 128, 128, 4, 3, 8),
        (256, 128, 128, 8, 3, 8),
        (256, 64, 128, 4, 4, 4),
        (256, 64, 128, 8, 4, 4),
        (64, 128, 64, 4, 4, 4),
        (64, 128, 64, 8, 4, 4),
        (64, 256, 128, 4, 4, 4),
        (64, 256, 128, 8, 4, 4),
        (64, 32, 64, 4, 5, 2),
    ] + get_configs_io_bound()
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "GROUP_SIZE_M": GROUP_SIZE_M,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        ) for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps in configs
    ]


def get_config_space_hopper_default(persistent: bool = False):
    if persistent:
        return [
            triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8, "EPILOGUE_SUBTILE": ep}, num_stages=s, num_warps=w) \
            for BM in [128] \
            for BN in [128, 256] \
            for BK in [64,128] \
            for s in ([3,4]) \
            for w in [4,8] \
            for ep in [True, False]
        ]

    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128, 256] \
        for BK in [64,128] \
        for s in ([3,4]) \
        for w in [4,8] \
    ]


def get_config_space_nt_h800(persistent=False):
    if persistent:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=8, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=8, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=8, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": True
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": True
                }, num_warps=4, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": False
                }, num_warps=8, num_stages=3),
        ]
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4,
                      num_stages=3),
    ]


def get_config_space_nn_h800(persistent=False):
    # for matmul_descriptor_persistent
    # TODO(houqi.1993) only for BF16/FP16
    if persistent:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=8, num_stages=3),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    True
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=4, num_stages=4),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4, "EPILOGUE_SUBTILE":
                    False
                }, num_warps=4, num_stages=4),
        ]

    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_warps=8,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}, num_warps=4,
                      num_stages=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}, num_warps=4,
                      num_stages=4),
    ]


def is_h800(device_id=0):
    return torch.cuda.get_device_name(device_id).find("NVIDIA H800") >= 0


def is_h100(device_id=0):
    return torch.cuda.get_device_name(device_id).find("NVIDIA H100") >= 0


def is_h20(device_id=0):
    # not H200
    device_name = torch.cuda.get_device_name(device_id)
    return device_name.find("NVIDIA H20") >= 0 and not device_name.find("NVIDIA H200") >= 0


def is_a100(device_id=0):
    # maybe A100 with nvlink, maybe 40GB
    return torch.cuda.get_device_name(device_id).find("NVIDIA A100") >= 0


def is_l20(device_id=0):
    return torch.cuda.get_device_name(device_id) == "NVIDIA L20"


def get_config_space(persistent=True, device_id=0):
    # if is_l20(device_id):  # not so accurate for L20 and so on. tune it yourself
    #     return get_config_space_ada()
    if is_a100(device_id):
        return get_config_space_a100()
    if is_h800(device_id) or is_h100(device_id):
        return get_config_space_nt_h800(persistent)

    # using default GEMM configs
    if torch.cuda.get_device_capability()[0] < 9:
        return get_config_space_default()
    return get_config_space_hopper_default(persistent)


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def prune_fn_by_shared_memory(config, A, *args, **kwargs) -> bool:
    itemsize = A.itemsize
    config = config["config"].all_kwargs()
    num_stages = config["num_stages"]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]
    shared_memory = (itemsize * BLOCK_SIZE_M * BLOCK_SIZE_K + itemsize * BLOCK_SIZE_N * BLOCK_SIZE_K) * num_stages
    return shared_memory < get_device_max_shared_memory_size(0)


def prune_fn_by_group_size_m(config, A, B, *args, **kwargs) -> bool:
    itemsize = A.itemsize
    config = config["config"].all_kwargs()
    M, K = A.shape
    _, N = B.shape
    tflops = 2 * M * K * N
    tflops_in_sec = tflops / 1e12 / get_tensorcore_tflops(A.dtype)
    memory_read_gb = (M * K * itemsize + K * N * itemsize) / 1e9
    memory_write_gb = M * N * itemsize / 1e9
    memory_read_in_sec = memory_read_gb / get_dram_gbps(A.dtype)
    memory_write_in_sec = memory_write_gb / get_dram_gbps(A.dtype)
    memory_in_sec = memory_read_in_sec + memory_write_in_sec

    GROUP_SIZE_M = config["GROUP_SIZE_M"]
    # GROUP_SIZE_M == 1 is for IO bound
    if tflops_in_sec / memory_in_sec > 2 and GROUP_SIZE_M == 1:
        return False

    return True


def prune_fn(config, A, B, *args, **kwargs) -> bool:
    return prune_fn_by_shared_memory(config, A, *args, **kwargs) and prune_fn_by_group_size_m(
        config, A, B, *args, **kwargs)


@triton_dist.tune.autotune(
    config_space=[{"config": c} for c in get_config_space(False)],
    key_fn=lambda *args, **kwargs: (to_hashable(args[0]), to_hashable(args[1])),
    prune_fn=prune_fn,
)
def matmul(a, b, config: triton.Config):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        **config.all_kwargs())
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma(a_desc, b_desc, c_desc,  #
                      M, N, K,  #
                      BLOCK_SIZE_M: tl.constexpr,  #
                      BLOCK_SIZE_N: tl.constexpr,  #
                      BLOCK_SIZE_K: tl.constexpr,  #
                      GROUP_SIZE_M: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      WARP_SPECIALIZE: tl.constexpr,  #
                      ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(k_tiles, warp_specialize=WARP_SPECIALIZE):
        offs_k = k * BLOCK_SIZE_K
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


@triton_dist.tune.autotune(
    config_space=[{"config": c} for c in get_config_space()],
    key_fn=lambda *args, **kwargs: (to_hashable(args[0]), to_hashable(args[1])),
    prune_fn=prune_fn_by_shared_memory,
)
def matmul_tma(a, b, config: triton.Config, warp_specialize: bool = False):
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    EPILOGUE_SUBTILE = config.kwargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2]
    else:
        c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), c_block_shape)

    matmul_kernel_tma[(triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        WARP_SPECIALIZE=warp_specialize,  #
        **config.all_kwargs())
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr,  #
                             M, N, K,  #
                             stride_am, stride_ak,  #
                             stride_bk, stride_bn,  #
                             stride_cm, stride_cn,  #
                             BLOCK_SIZE_M: tl.constexpr,  #
                             BLOCK_SIZE_N: tl.constexpr,  #
                             BLOCK_SIZE_K: tl.constexpr,  #
                             GROUP_SIZE_M: tl.constexpr,  #
                             NUM_SMS: tl.constexpr,  #
                             ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if (c_ptr.dtype.element_ty == tl.float8e4nv):
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


@triton_dist.tune.autotune(
    config_space=[{"config": c} for c in get_config_space()],
    key_fn=lambda *args, **kwargs: (to_hashable(args[0]), to_hashable(args[1])),
    prune_fn=prune_fn_by_shared_memory,
)
def matmul_persistent(a, b, config: triton.Config):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        **config.all_kwargs(),
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 FP8_OUTPUT: tl.constexpr,  #
                                 EPILOGUE_SUBTILE: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 WARP_SPECIALIZE: tl.constexpr,  #
                                 ):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Enable warp specialization to leverage async warp scheduling in the GPU.
    # FIXME: This only works on Blackwell right now. On older GPUs, this will
    # use software pipelining.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc.store([offs_am_c, offs_bn_c], accumulator)


@triton_dist.tune.autotune(
    config_space=[{"config": c} for c in get_config_space(persistent=True)],
    key_fn=lambda *args, **kwargs: (to_hashable(args[0]), to_hashable(args[1])),
    prune_fn=prune_fn_by_shared_memory,
)
def matmul_tma_persistent(a, b, config: triton.Config, warp_specialize: bool = False):
    # Check constraints.
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]
    EPILOGUE_SUBTILE = config.kwargs.get("EPILOGUE_SUBTILE", False)
    if EPILOGUE_SUBTILE:
        c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2]
    else:
        c_block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), c_block_shape)

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        BLOCK_SIZE_M = META["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = META["BLOCK_SIZE_N"]
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc,  #
        M, N, K,  #
        FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        **config.all_kwargs())
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr,  #
                                        BLOCK_SIZE_N: tl.constexpr,  #
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        GROUP_SIZE_M: tl.constexpr,  #
                                        EPILOGUE_SUBTILE: tl.constexpr,  #
                                        NUM_SMS: tl.constexpr,  #
                                        WARP_SPECIALIZE: tl.constexpr,  #
                                        ):
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

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


@triton_dist.tune.autotune(
    config_space=[{"config": c} for c in get_config_space(persistent=True)],
    key_fn=lambda *args, **kwargs: (to_hashable(args[0]), to_hashable(args[1])),
    prune_fn=prune_fn_by_shared_memory,
)
def matmul_descriptor_persistent(a, b, config: triton.Config, warp_specialize: bool = False):
    # Check constraints.
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a, b, c,  #
        M, N, K,  #
        NUM_SMS=NUM_SMS,  #
        WARP_SPECIALIZE=warp_specialize,  #
        **config.all_kwargs())
    return c
