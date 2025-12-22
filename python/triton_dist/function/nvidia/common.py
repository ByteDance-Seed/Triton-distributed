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
import os
from dataclasses import dataclass

from triton_dist.kernels.nvidia.group_gemm import (
    GROUP_GEMM_BLOCK_SIZE_M, )
from triton_dist.layers.nvidia.ep_a2a_fused_layer import EpAll2AllFusedOp

from typing import Optional
try:
    from torch.amp import custom_bwd as torch_custom_bwd
    from torch.amp import custom_fwd as torch_custom_fwd

    CUSTOM_FWD_BWD_EXTRA_KWARGS = {"device_type": "cuda"}
except ImportError:
    from torch.cuda.amp import custom_bwd as torch_custom_bwd
    from torch.cuda.amp import custom_fwd as torch_custom_fwd

    CUSTOM_FWD_BWD_EXTRA_KWARGS = {}


# Below goes `custom_{fwd,bwd}` decorator that's compatible with both PyTorch 2.1 and 2.4
def custom_fwd(*args, **kwargs):
    return torch_custom_fwd(*args, **kwargs, **CUSTOM_FWD_BWD_EXTRA_KWARGS)


def custom_bwd(*args, **kwargs):
    return torch_custom_bwd(*args, **kwargs, **CUSTOM_FWD_BWD_EXTRA_KWARGS)


# Global context
# Ditron uses NVSHMEM, which should only be initialized once per process.
# We use global variables to store the context.

MAX_TOKENS_PER_RANK = None
DITRON_EP_STREAM = None

# no split mbs global context
triton_dist_ep_op = None

# split mbs global context
triton_dist_ep_op1 = None
triton_dist_ep_op2 = None
triton_dist_ep_op_bwd = None

DITRON_EP_STREAM_1 = None
DITRON_EP_STREAM_2 = None
fwd_dispatch1_event = None
bwd_dispatch_event = None
bwd_combine_event = None

# SM-level profiling context
PROFILE_DITRONT_MOE_FWD_DISPATCH = False
PROFILE_DITRONT_MOE_FWD_COMBINE = False
PROFILE_DITRONT_MOE_BWD_DISPATCH = False
PROFILE_DITRONT_MOE_BWD_COMBINE = False

# Profile output directory for perfetto traces
DITRON_PROFILE_OUTPUT_DIR = "prof/mega"


def set_triton_dist_moe_profile_enabled(
    enabled: bool,
    fwd_dispatch: bool = True,
    fwd_combine: bool = True,
    bwd_dispatch: bool = True,
    bwd_combine: bool = True,
    output_dir: str = None,
) -> None:
    """
    Enable or disable Ditron MoE profiling.
    
    Args:
        enabled: Whether to enable profiling.
        fwd_dispatch: Whether to profile forward dispatch.
        fwd_combine: Whether to profile forward combine.
        bwd_dispatch: Whether to profile backward dispatch.
        bwd_combine: Whether to profile backward combine.
        output_dir: Directory to save perfetto traces. If None, uses default "prof/mega".
    """
    global PROFILE_DITRONT_MOE_FWD_DISPATCH, PROFILE_DITRONT_MOE_FWD_COMBINE
    global PROFILE_DITRONT_MOE_BWD_DISPATCH, PROFILE_DITRONT_MOE_BWD_COMBINE
    global DITRON_PROFILE_OUTPUT_DIR

    PROFILE_DITRONT_MOE_FWD_DISPATCH = enabled and fwd_dispatch
    PROFILE_DITRONT_MOE_FWD_COMBINE = enabled and fwd_combine
    PROFILE_DITRONT_MOE_BWD_DISPATCH = enabled and bwd_dispatch
    PROFILE_DITRONT_MOE_BWD_COMBINE = enabled and bwd_combine

    if output_dir is not None:
        DITRON_PROFILE_OUTPUT_DIR = output_dir


def get_triton_dist_moe_profile_enabled() -> dict:
    """
    Get current Ditron MoE profiling state.
    
    Returns:
        Dictionary with profiling flags for each operation.
    """
    return {
        "fwd_dispatch": PROFILE_DITRONT_MOE_FWD_DISPATCH,
        "fwd_combine": PROFILE_DITRONT_MOE_FWD_COMBINE,
        "bwd_dispatch": PROFILE_DITRONT_MOE_BWD_DISPATCH,
        "bwd_combine": PROFILE_DITRONT_MOE_BWD_COMBINE,
        "output_dir": DITRON_PROFILE_OUTPUT_DIR,
    }


def get_triton_dist_profile_output_dir() -> str:
    """Get the directory where Ditron profile traces are saved."""
    return DITRON_PROFILE_OUTPUT_DIR


class TritonDistEpContext:

    def __init__(self, ep_group, ep_op, ep_stream, ep_events, num_experts_per_rank, MAX_M):
        self.ep_group = ep_group
        self.ep_op = ep_op
        self.max_num_tiles = (MAX_M + GROUP_GEMM_BLOCK_SIZE_M - 1) // GROUP_GEMM_BLOCK_SIZE_M + num_experts_per_rank
        self.split_size_cum_per_expert = torch.empty([num_experts_per_rank], dtype=torch.int32, device="cuda")
        self.expert_ids = torch.empty([self.max_num_tiles], dtype=torch.int32, device="cuda")
        self.split_size_cum = torch.empty([self.max_num_tiles], dtype=torch.int32, device="cuda")
        self.tile_num = torch.empty([self.max_num_tiles], dtype=torch.int32, device="cuda")
        self.tile_num_cum = torch.empty([self.max_num_tiles], dtype=torch.int32, device="cuda")
        self.expert_tile_offset = torch.empty([num_experts_per_rank], dtype=torch.int32, device="cuda")
        self.num_tiles_total = torch.empty([1], dtype=torch.int32, device="cuda")
        self.ep_a2a_layout_desc = None
        self.triton_dist_ep_stream = ep_stream
        self.triton_dist_ep_events = ep_events
        self.swiglu_ctx = None


def triton_dist_ep_op_initialized(ep_implementation: str = "mega"):
    global triton_dist_ep_op
    global triton_dist_ep_op1
    global triton_dist_ep_op2
    if ep_implementation in ["mega", "mega_recomp"]:
        return triton_dist_ep_op is not None
    elif ep_implementation == "split_mbs":
        return triton_dist_ep_op1 is not None and triton_dist_ep_op2 is not None
    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['mega', 'mega_recomp', 'split_mbs']")


def init_triton_dist_ep_op(ep_group, max_tokens_per_rank, hidden_size, topk, ep_rank, num_experts, ep_size,
                           dtype=torch.bfloat16, weight_dtype=torch.float32, num_sm=8, sm_margin=0, num_buffers=1,
                           capacity=4.0, ep_implementation: str = "mega",  # ["mega", "mega_recomp", "split_mbs"]
                           ):
    global triton_dist_ep_op
    global triton_dist_ep_op1
    global triton_dist_ep_op2
    global triton_dist_ep_op_bwd
    global MAX_TOKENS_PER_RANK
    global DITRON_EP_STREAM
    global DITRON_EP_STREAM_1
    global DITRON_EP_STREAM_2
    global fwd_dispatch1_event
    global bwd_dispatch_event
    global bwd_combine_event

    NVSHMEM_SYMMETRIC_SIZE = os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "-1")
    if NVSHMEM_SYMMETRIC_SIZE.endswith("g"):
        NVSHMEM_SYMMETRIC_SIZE = int(NVSHMEM_SYMMETRIC_SIZE[:-1]) * 1e9
    elif NVSHMEM_SYMMETRIC_SIZE.endswith("m"):
        NVSHMEM_SYMMETRIC_SIZE = int(NVSHMEM_SYMMETRIC_SIZE[:-1]) * 1e6
    elif NVSHMEM_SYMMETRIC_SIZE.endswith("k"):
        NVSHMEM_SYMMETRIC_SIZE = int(NVSHMEM_SYMMETRIC_SIZE[:-1]) * 1e3
    else:
        NVSHMEM_SYMMETRIC_SIZE = int(NVSHMEM_SYMMETRIC_SIZE)

    triton_dist_ep_ops = []

    if ep_implementation in ["mega", "mega_recomp"]:
        if triton_dist_ep_op is not None:
            print(f"triton_dist_ep_op already initialized at rank {ep_group.rank()}")
            return
        DITRON_EP_STREAM = torch.cuda.Stream()
        MAX_TOKENS_PER_RANK = max_tokens_per_rank
        triton_dist_ep_op = EpAll2AllFusedOp(ep_group, max_tokens_per_rank, hidden_size, topk, ep_rank, num_experts,
                                             min(8, ep_size), ep_size, dtype=dtype, weight_dtype=weight_dtype,
                                             num_sm=num_sm, sm_margin=sm_margin, duplicate_comm_buffer=num_buffers,
                                             capacity=capacity, FWD_GEMM_BLOCK_SIZE_N=256,
                                             need_reversed_token_scatter_idx=True, lazy=True)

        # Print nvshmem memory requirement before allocation
        if ep_rank == 0:
            print(f"[EpAll2AllOp] nvshmem memory required: {triton_dist_ep_op.get_nvshmem_size_mb():.2f} MB "
                  f"({triton_dist_ep_op.get_nvshmem_size_gb():.4f} GB)")
            triton_dist_ep_op.print_nvshmem_breakdown()

        total_nvshmem = triton_dist_ep_op.get_nvshmem_size()
        total_nvshmem_mb = triton_dist_ep_op.get_nvshmem_size_mb()
        total_nvshmem_gb = triton_dist_ep_op.get_nvshmem_size_gb()

        # Actually allocate nvshmem memory
        triton_dist_ep_ops.append(triton_dist_ep_op)
    elif ep_implementation == "split_mbs":
        if triton_dist_ep_op1 is not None:
            print(f"triton_dist_ep_op already initialized at rank {ep_group.rank()}")
            return
        DITRON_EP_STREAM = torch.cuda.Stream()
        DITRON_EP_STREAM_1 = torch.cuda.Stream()
        DITRON_EP_STREAM_2 = torch.cuda.Stream()
        fwd_dispatch1_event = torch.cuda.Event()
        bwd_dispatch_event = torch.cuda.Event()
        bwd_combine_event = torch.cuda.Event()

        MAX_TOKENS_PER_RANK = max_tokens_per_rank // 2
        triton_dist_ep_op1 = EpAll2AllFusedOp(ep_group,
                                              max_tokens_per_rank // 2, hidden_size, topk, ep_rank, num_experts,
                                              min(8, ep_size), ep_size, dtype=dtype, weight_dtype=weight_dtype,
                                              num_sm=num_sm, duplicate_comm_buffer=num_buffers, capacity=capacity,
                                              lazy=True)
        triton_dist_ep_op2 = EpAll2AllFusedOp(ep_group,
                                              max_tokens_per_rank // 2, hidden_size, topk, ep_rank, num_experts,
                                              min(8, ep_size), ep_size, dtype=dtype, weight_dtype=weight_dtype,
                                              num_sm=num_sm, duplicate_comm_buffer=num_buffers, capacity=capacity,
                                              lazy=True)

        # Print nvshmem memory requirement before allocation
        total_nvshmem = triton_dist_ep_op1.get_nvshmem_size() + triton_dist_ep_op2.get_nvshmem_size()
        total_nvshmem_mb = triton_dist_ep_op1.get_nvshmem_size_mb() + triton_dist_ep_op2.get_nvshmem_size_mb()
        total_nvshmem_gb = triton_dist_ep_op1.get_nvshmem_size_gb() + triton_dist_ep_op2.get_nvshmem_size_gb()
        if ep_rank == 0:
            print(f"[EpAll2AllOp] nvshmem memory required (op1): {triton_dist_ep_op1.get_nvshmem_size_mb():.2f} MB")
            print(f"[EpAll2AllOp] nvshmem memory required (op2): {triton_dist_ep_op2.get_nvshmem_size_mb():.2f} MB")
            print(f"[EpAll2AllOp] total nvshmem memory required: {total_nvshmem_mb:.2f} MB ({total_nvshmem_gb:.4f} GB)")

        # Actually allocate nvshmem memory
        triton_dist_ep_ops.append(triton_dist_ep_op1)
        triton_dist_ep_ops.append(triton_dist_ep_op2)

    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['triton_dist', 'mega_recomp', 'split_mbs']")

    if NVSHMEM_SYMMETRIC_SIZE == -1 or total_nvshmem > NVSHMEM_SYMMETRIC_SIZE:
        print(f"[EpAll2AllOp] NVSHMEM_SYMMETRIC_SIZE is too small, required: {total_nvshmem_mb:.2f} MB "
              f"({total_nvshmem_gb:.4f} GB), but NVSHMEM_SYMMETRIC_SIZE is {NVSHMEM_SYMMETRIC_SIZE} bytes")
        headroom = 500000000  # 500MB
        aligment_size = 100000000  # 100MB
        total = int((total_nvshmem + headroom) // aligment_size * aligment_size)
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(total)
        if ep_rank == 0:
            print(f"[EpAll2AllOp] NVSHMEM_SYMMETRIC_SIZE is updated to {total} bytes")

    for ops in triton_dist_ep_ops:
        ops.sync()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    torch.distributed.barrier(ep_group)


def deinit_triton_dist_ep_op(ep_implementation: str = "mega"):
    global triton_dist_ep_op
    global MAX_TOKENS_PER_RANK
    global triton_dist_ep_op_bwd
    global triton_dist_ep_op1
    global triton_dist_ep_op2

    if ep_implementation in ["mega", "mega_recomp"]:
        MAX_TOKENS_PER_RANK = None
        triton_dist_ep_op = None
    elif ep_implementation == "split_mbs":
        MAX_TOKENS_PER_RANK = None
        triton_dist_ep_op1 = None
        triton_dist_ep_op2 = None
    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['triton_dist', 'mega_recomp', 'split_mbs']")


def init_triton_dist_ep_ctx(
    ep_group,
    topk,
    num_experts,
    ep_implementation: str = "mega",  # ["mega", "mega_recomp", "split_mbs"]
    mbs_idx=0,
):
    global triton_dist_ep_op
    global triton_dist_ep_op1
    global triton_dist_ep_op2
    global triton_dist_ep_op_bwd
    global MAX_TOKENS_PER_RANK
    global DITRON_EP_STREAM

    if ep_implementation in ["mega", "mega_recomp"]:
        assert triton_dist_ep_op is not None, "Please initialize triton_dist_ep_op first."
        assert MAX_TOKENS_PER_RANK is not None, "Please initialize triton_dist_ep_op first."
        assert num_experts % ep_group.size() == 0
        triton_dist_ep_ctx = TritonDistEpContext(ep_group, triton_dist_ep_op, DITRON_EP_STREAM, {},
                                                 num_experts // ep_group.size(),
                                                 MAX_TOKENS_PER_RANK * topk * ep_group.size())
        return triton_dist_ep_ctx
    elif ep_implementation == "split_mbs":
        assert triton_dist_ep_op1 is not None, "Please initialize triton_dist_ep_op1 first."
        assert triton_dist_ep_op2 is not None, "Please initialize triton_dist_ep_op2 first."
        assert MAX_TOKENS_PER_RANK is not None, "Please initialize triton_dist_ep_op first."
        assert num_experts % ep_group.size() == 0
        if mbs_idx == 0:
            assert triton_dist_ep_op1 is not None, "Please initialize triton_dist_ep_op1 first."
            triton_dist_ep_op = triton_dist_ep_op1
        elif mbs_idx == 1:
            assert triton_dist_ep_op2 is not None, "Please initialize triton_dist_ep_op2 first."
            triton_dist_ep_op = triton_dist_ep_op2
        else:
            assert triton_dist_ep_op_bwd is not None, "Please initialize triton_dist_ep_op_bwd first."
            triton_dist_ep_op = triton_dist_ep_op_bwd
        ep_events = {
            "fwd_dispatch": fwd_dispatch1_event,
            "bwd_dispatch": bwd_dispatch_event,
            "bwd_combine": bwd_combine_event,
        }
        triton_dist_ep_ctx = TritonDistEpContext(ep_group, triton_dist_ep_op, DITRON_EP_STREAM, ep_events,
                                                 num_experts // ep_group.size(),
                                                 MAX_TOKENS_PER_RANK * topk * ep_group.size())
        return triton_dist_ep_ctx
    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['triton_dist', 'mega_recomp', 'split_mbs']")


def get_ep_capacity(ep_implementation: str = "mega"):
    global triton_dist_ep_op
    global triton_dist_ep_op1
    global triton_dist_ep_op2
    global triton_dist_ep_op_bwd

    assert triton_dist_ep_op_initialized(ep_implementation), "Please initialize triton_dist_ep_op first."

    if ep_implementation in ["mega", "mega_recomp"]:
        return triton_dist_ep_op.capacity
    elif ep_implementation == "split_mbs":
        assert triton_dist_ep_op1.capacity == triton_dist_ep_op2.capacity, "Split mbs capacity must be the same"
        return triton_dist_ep_op1.capacity
    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['mega', 'mega_recomp', 'split_mbs']")


def get_triton_dist_ep_stream(idx=0):
    global DITRON_EP_STREAM
    global DITRON_EP_STREAM_1
    global DITRON_EP_STREAM_2
    if idx == 0:
        if DITRON_EP_STREAM is None:
            print("Warning: DITRON_EP_STREAM is not initialized.")
        return DITRON_EP_STREAM
    elif idx == 1:
        if DITRON_EP_STREAM_1 is None:
            print("Warning: DITRON_EP_STREAM_1 is not initialized.")
        return DITRON_EP_STREAM_1
    elif idx == 2:
        if DITRON_EP_STREAM_2 is None:
            print("Warning: DITRON_EP_STREAM_2 is not initialized.")
        return DITRON_EP_STREAM_2
    else:
        raise ValueError(f"Invalid idx: {idx}, expected: [0, 1, 2]")


def get_triton_dist_ep_op(idx=0):
    global triton_dist_ep_op
    global triton_dist_ep_op1
    global triton_dist_ep_op2
    if idx == 0:
        return triton_dist_ep_op
    elif idx == 1:
        return triton_dist_ep_op1
    elif idx == 2:
        return triton_dist_ep_op2
    else:
        raise ValueError(f"Invalid idx: {idx}, expected: [0, 1, 2]")


@dataclass
class MoEOptimConfig:
    num_build_sms: int
    num_copy_sms: int
    num_group_gemm_warps: int
    num_dispatch_warps: int
    num_combine_warps: int
    num_dispatch_sms: int
    num_tail_sms_in_dispatch: int
    num_combine_sms: int
    num_reduce_sms_in_combine: int
    dispatch_use_block_wise_barrier: bool


def get_moe_optim_config(use_mega: bool = False, is_forward: bool = True):
    max_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if is_forward:
        if max_sms > 78:  # for H800
            if use_mega:
                return MoEOptimConfig(
                    num_build_sms=8,
                    num_copy_sms=max_sms,
                    num_group_gemm_warps=8,
                    num_dispatch_warps=16,
                    num_combine_warps=32,
                    num_dispatch_sms=80,
                    num_tail_sms_in_dispatch=32,
                    num_combine_sms=80,
                    num_reduce_sms_in_combine=80,
                    dispatch_use_block_wise_barrier=True,
                )
            else:
                return MoEOptimConfig(
                    num_build_sms=8,
                    num_copy_sms=max_sms,
                    num_group_gemm_warps=8,
                    num_dispatch_warps=32,
                    num_combine_warps=32,
                    num_dispatch_sms=64,
                    num_tail_sms_in_dispatch=0,
                    num_combine_sms=64,
                    num_reduce_sms_in_combine=0,
                    dispatch_use_block_wise_barrier=False,
                )
        else:  # for H20
            print("Warning: H20 is not tuned for forward.")
            return MoEOptimConfig(
                num_build_sms=8,
                num_copy_sms=32,
                num_group_gemm_warps=32,
                num_dispatch_warps=32,
                num_combine_warps=32,
                num_dispatch_sms=64,
                num_tail_sms_in_dispatch=0,
                num_combine_sms=64,
                num_reduce_sms_in_combine=0,
                dispatch_use_block_wise_barrier=False,
            )
    else:  # backward
        if max_sms > 78:  # for H800
            if use_mega:
                return MoEOptimConfig(
                    num_build_sms=8,
                    num_copy_sms=max_sms,
                    num_group_gemm_warps=8,
                    num_dispatch_warps=32,
                    num_combine_warps=16,
                    num_dispatch_sms=64,
                    num_tail_sms_in_dispatch=16,
                    num_combine_sms=64,
                    num_reduce_sms_in_combine=100,
                    dispatch_use_block_wise_barrier=True,
                )
            else:
                return MoEOptimConfig(
                    num_build_sms=8,
                    num_copy_sms=max_sms,
                    num_group_gemm_warps=8,
                    num_dispatch_warps=32,
                    num_combine_warps=32,
                    num_dispatch_sms=64,
                    num_tail_sms_in_dispatch=0,
                    num_combine_sms=64,
                    num_reduce_sms_in_combine=0,
                    dispatch_use_block_wise_barrier=False,
                )
        else:  # for H20
            print("Warning: H20 is not tuned for backward.")
            return MoEOptimConfig(
                num_build_sms=8,
                num_copy_sms=32,
                num_group_gemm_warps=32,
                num_dispatch_warps=32,
                num_combine_warps=32,
                num_dispatch_sms=64,
                num_tail_sms_in_dispatch=0,
                num_combine_sms=64,
                num_reduce_sms_in_combine=0,
                dispatch_use_block_wise_barrier=False,
            )
