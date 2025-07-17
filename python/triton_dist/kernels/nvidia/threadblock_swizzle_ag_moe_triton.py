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

import argparse
import logging
import random
from typing import Any, List, Tuple

import numpy as np
import torch

import triton
import triton.language as tl
from triton.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.kernels.nvidia.common_ops import next_power_of_2, bisect_right_kernel


@triton.jit
def local_segment_to_local_stage(segment, rank, LOCAL_TP_SIZE: tl.constexpr):
    return (segment - rank + LOCAL_TP_SIZE) % LOCAL_TP_SIZE


@triton.jit
def segment_to_stage(segment, rank, TP_SIZE: tl.constexpr, LOCAL_TP_SIZE: tl.constexpr):
    NNODES: tl.constexpr = TP_SIZE // LOCAL_TP_SIZE
    snode = segment // LOCAL_TP_SIZE
    node = rank // LOCAL_TP_SIZE
    off_node = (snode - node + NNODES) % NNODES
    return off_node * LOCAL_TP_SIZE + local_segment_to_local_stage(segment, rank, LOCAL_TP_SIZE)


@triton.jit
def get_tile_stage(
    tiled_m,
    rank,
    TP_SIZE: tl.constexpr,
    LOCAL_TP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    ntokens_by_rank_acc_ptr,
):
    ntokens_this_expert = tl.load(ntokens_by_rank_acc_ptr + TP_SIZE - 1)
    global_m_start = tl.where(rank == 0, 0, tl.load(ntokens_by_rank_acc_ptr + rank - 1))
    global_tiled_m_start = tl.cdiv(global_m_start, BLOCK_SIZE_M)
    m_start = tiled_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, ntokens_this_expert) - 1
    segment_start = bisect_right_kernel(ntokens_by_rank_acc_ptr, m_start, TP_SIZE)
    segment_end = bisect_right_kernel(ntokens_by_rank_acc_ptr, m_end, TP_SIZE)
    stage = segment_to_stage(segment_end, rank, TP_SIZE, LOCAL_TP_SIZE)

    stage = tl.where(
        (tiled_m == global_tiled_m_start - 1) & (global_tiled_m_start % BLOCK_SIZE_M != 0),
        TP_SIZE - 1,
        stage,
    )  # TODO(houqi.1993) make it easier
    return stage, segment_start, segment_end


@triton.jit
def bisect_right_with_offset_kernel(sorted_ptr, values, N: tl.constexpr):
    """
    index = bisect(sorted_ptr, values, N)
    off = sorted_ptr[index - 1] : let suppose values[-1] = 0
    remainder = values - off

    It's expected that values in [0, sorted_ptr[N-1])
    """
    index = bisect_right_kernel(sorted_ptr, values, N)
    off = tl.where(index == 0, 0, tl.load(sorted_ptr + index - 1))
    reminder = values - off
    return index, off, reminder


@triton.jit(do_not_specialize=["rank"])
def threadblock_swizzle_ag_moe_kernel(
    # input
    ntokens_by_rank_by_expert_ptr,
    # output
    expert_id_ptr,
    tiled_m_ptr,
    segment_start_ptr,
    segment_end_ptr,
    ntiles_ptr,
    # workspace buffer
    ntokens_by_expert_by_rank_acc_ptr,
    ntiles_by_expert_acc_ptr,
    ntiles_by_expert_by_stage_ptr,
    ntiles_by_expert_by_stage_acc_ptr,
    rank,
    N_EXPERTS: tl.constexpr,
    TP_SIZE: tl.constexpr,
    LOCAL_TP_SIZE: tl.constexpr,
    NTILES_NEXT_POW_OF_2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    """
    tile_index = g(expert_id, stage, index): if tile_index is grouped by expert then stage

    but how to map (stage, index) => tile_index_in_expert?

    first map tile_index_in_expert => (stage, off_in_expert_by_stage, index_in_expert_in_stage)

    tile_index => (expert_id, tile_index_in_expert)                 : ntokens grouped by expert_id by rank
               => (expert_id, stage)                                : get_stage_id
               => (expert_id, off_in_expert_by_stage)               : cumsum by stage
               => (expert_id, off_in_expert_by_stage, index_in_expert_in_stage)  : atomic_add for index_in_expert_in_stage
               => tile_index_new : by function g

    tile_index -> (expert_id, rank_offset) grouped by (expert_id, rank)
    rank_offset -> segment_start, segment_end, stage

    so we can remap tile_index as:
    tile_index_new = g(f(tile_index))
    """
    thread_idx = tid(0)
    N_EXPERTS_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(N_EXPERTS)
    TP_SIZE_NEXT_POW_OF_2: tl.constexpr = next_power_of_2(TP_SIZE)
    offs_by_expert = tl.arange(0, N_EXPERTS_NEXT_POW_OF_2)
    mask_by_expert = offs_by_expert < N_EXPERTS
    offs_by_rank = tl.arange(0, TP_SIZE_NEXT_POW_OF_2)
    mask_by_rank = (offs_by_rank < TP_SIZE)
    offs_by_expert_by_rank = offs_by_expert[:, None] * TP_SIZE + offs_by_rank[None, :]
    mask_by_expert_by_rank = mask_by_expert[:, None] & mask_by_rank[None, :]
    offs_by_rank_by_expert = offs_by_rank[:, None] * N_EXPERTS + offs_by_expert[None, :]
    mask_by_rank_by_expert = mask_by_rank[:, None] & mask_by_expert[None, :]
    ntokens_by_rank_by_expert = tl.load(ntokens_by_rank_by_expert_ptr + offs_by_rank_by_expert,
                                        mask=mask_by_rank_by_expert)

    ntokens_by_expert_by_rank = ntokens_by_rank_by_expert.T
    ntokens_by_expert_by_rank_acc = tl.cumsum(ntokens_by_expert_by_rank, axis=1)
    ntokens_by_expert = tl.sum(ntokens_by_rank_by_expert, axis=0)
    ntiles_by_expert = tl.cdiv(ntokens_by_expert, BLOCK_SIZE_M)
    ntiles_by_expert_acc = tl.cumsum(ntiles_by_expert, axis=0)
    ntiles = tl.sum(ntiles_by_expert)

    tl.store(ntokens_by_expert_by_rank_acc_ptr + offs_by_expert_by_rank, ntokens_by_expert_by_rank_acc,
             mask=mask_by_expert_by_rank)
    tl.store(ntiles_by_expert_acc_ptr + offs_by_expert, ntiles_by_expert_acc)

    # # for each tiled_m in expert eid => stage id / segment_start / segmeng_end / tiled_m
    tile_index = tl.arange(0, NTILES_NEXT_POW_OF_2)
    mask_tile_idx = tile_index < ntiles

    __syncthreads()
    # tile_index -> (expert_id, offset_by_expert, tile_index_in_expert) -> stage, segment_start, segment_end
    expert_id, off_by_expert, off_in_expert = bisect_right_with_offset_kernel(ntiles_by_expert_acc_ptr, tile_index,
                                                                              N_EXPERTS)

    stage, segment_start, segment_end = get_tile_stage(
        off_in_expert,
        rank,
        TP_SIZE,
        LOCAL_TP_SIZE,
        BLOCK_SIZE_M,
        ntokens_by_expert_by_rank_acc_ptr + expert_id * TP_SIZE,
    )

    # histogram by expert by stage
    tl.store(ntiles_by_expert_by_stage_ptr + offs_by_expert_by_rank, 0, mask=mask_by_expert_by_rank)
    __syncthreads()
    off_in_expert_in_stage = tl.atomic_add(ntiles_by_expert_by_stage_ptr + expert_id * TP_SIZE + stage, 1,
                                           sem="relaxed", scope="gpu", mask=mask_tile_idx)
    __syncthreads()

    # do some cumsum
    ntiles_by_expert_by_stage = tl.load(ntiles_by_expert_by_stage_ptr + offs_by_expert_by_rank,
                                        mask=mask_by_expert_by_rank, other=0)
    ntiles_by_expert_by_stage_acc = tl.cumsum(ntiles_by_expert_by_stage, axis=1)
    __syncthreads()
    tl.store(
        ntiles_by_expert_by_stage_acc_ptr + offs_by_expert_by_rank,
        ntiles_by_expert_by_stage_acc,
        mask=mask_by_expert_by_rank,
    )
    __syncthreads()

    off_in_expert_by_stage = tl.where(
        stage == 0,
        0,
        tl.load(ntiles_by_expert_by_stage_acc_ptr + expert_id * TP_SIZE + stage - 1, mask=mask_tile_idx, other=0),
    )

    tile_index_by_expert_by_stage = off_by_expert + off_in_expert_by_stage + off_in_expert_in_stage
    __syncthreads()

    tl.store(expert_id_ptr + tile_index_by_expert_by_stage, expert_id, mask=mask_tile_idx)
    tl.store(tiled_m_ptr + tile_index_by_expert_by_stage, tile_index, mask=mask_tile_idx)
    tl.store(segment_start_ptr + tile_index_by_expert_by_stage, segment_start, mask=mask_tile_idx)
    tl.store(segment_end_ptr + tile_index_by_expert_by_stage, segment_end, mask=mask_tile_idx)
    thread_idx = tid(0)
    if thread_idx == 0:
        tl.store(ntiles_ptr, ntiles)
    if DEBUG and thread_idx < ntiles:
        print("expert_id", expert_id)


def transpose2d(arr_2d: List[List[Any]], ) -> List[Any]:
    dim0 = len(arr_2d)
    dim1 = len(arr_2d[0])
    for n in arr_2d:
        assert len(n) == dim1

    flatten = [tiles for arr_1d in arr_2d for tiles in arr_1d]
    reshaped = [flatten[rank::dim1] for rank in range(dim1)]
    assert len(reshaped) == dim1
    for n in reshaped:
        assert len(n) == dim0
    return reshaped


def threadblock_swizzle_ag_moe_triton(
    rank,
    N_EXPERTS: int,
    TP_SIZE: int,
    LOCAL_TP_SIZE: int,
    BLOCK_SIZE_M: int,
    ntokens_by_rank_by_expert: np.ndarray,
    verbose: bool = False,
):
    ntokens_by_expert_by_rank = ntokens_by_rank_by_expert.T
    ntokens_by_expert = ntokens_by_expert_by_rank.sum(axis=1)
    ntiles_by_expert = (ntokens_by_expert + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    ntiles = ntiles_by_expert.sum()
    # print("ntokens_by_expert", ntokens_by_expert)
    # print("ntokens_by_expert_by_rank", ntokens_by_expert_by_rank)
    # print("ntiles_by_expert", ntiles_by_expert)
    segment_start = torch.empty((ntiles, ), dtype=torch.int32, device="cuda")
    segment_end = torch.empty((ntiles, ), dtype=torch.int32, device="cuda")
    expert_idx = torch.empty((ntiles, ), dtype=torch.int32, device="cuda")
    tile_index = torch.empty((ntiles, ), dtype=torch.int32, device="cuda")
    ntiles_gpu = torch.empty((1, ), dtype=torch.int32, device="cuda")

    ntokens_by_rank_by_expert = torch.from_numpy(ntokens_by_rank_by_expert).cuda()
    ntokens_by_expert_by_rank_acc = torch.empty((N_EXPERTS, TP_SIZE), dtype=torch.int32, device="cuda")
    ntiles_by_expert_acc = torch.empty((N_EXPERTS, ), dtype=torch.int32, device="cuda")
    ntiles_by_expert_by_stage = torch.empty((N_EXPERTS, TP_SIZE), dtype=torch.int32,
                                            device="cuda")  # this will be used as counter. zero before use.
    ntiles_by_expert_by_stage_acc = torch.empty((N_EXPERTS, TP_SIZE), dtype=torch.int32, device="cuda")

    # print("ntiles", ntiles)

    NTILES = int(triton.next_power_of_2(ntiles))

    threadblock_swizzle_ag_moe_kernel[(1, )](
        ntokens_by_rank_by_expert,
        # output
        expert_idx,
        tile_index,
        segment_start,
        segment_end,
        ntiles_gpu,
        # workspace buffer
        ntokens_by_expert_by_rank_acc,
        ntiles_by_expert_acc,
        ntiles_by_expert_by_stage,
        ntiles_by_expert_by_stage_acc,
        rank,
        N_EXPERTS,
        TP_SIZE,
        LOCAL_TP_SIZE,
        NTILES,
        BLOCK_SIZE_M,
        DEBUG=verbose,
        num_warps=max(NTILES, 32) // 32,
    )
    if verbose:
        print("ntokens_by_expert_by_rank_acc", ntokens_by_expert_by_rank_acc)
        print("ntiles_by_expert_acc", ntiles_by_expert_acc)
        print("ntiles_by_expert_by_stage", ntiles_by_expert_by_stage)
        print("ntiles_by_expert_by_stage_acc", ntiles_by_expert_by_stage_acc)

    return expert_idx, tile_index


def check_swizzled(swizzled: List[Tuple[int, int]], ntokens_per_rank_per_expert, BLOCK_SIZE_M):
    # check each expert all tiles is calculated

    ntokens_per_expert_per_rank = transpose2d(ntokens_per_rank_per_expert)
    nexperts = len(ntokens_per_expert_per_rank)
    for expert_id in range(nexperts):
        tokens_this_ep = sum(ntokens_per_expert_per_rank[expert_id])
        ntiles_this_ep = triton.cdiv(tokens_this_ep, BLOCK_SIZE_M)
        tiled_m = [tiled_m for (eid, tiled_m) in swizzled if eid == expert_id]
        tiled_m.sort()
        assert tiled_m == list(range(ntiles_this_ep)), (
            expert_id,
            tiled_m,
            ntiles_this_ep,
        )


def generate_ntokens_per_rank_per_expert_uniform(ntokens_per_rank, nexperts, TP_SIZE):
    """
    [
        [ntokens(r0, e0), ntokens(r0, e1), ...],
        [ntokens(r1, e0), ntokens(r1, e1), ...],
        ...
        [ntokens(r7, e0), ntokens(r7, e1), ...],
    ]
    """
    assert ntokens_per_rank % nexperts == 0
    ntokens_per_expert = ntokens_per_rank // nexperts
    return np.array([[ntokens_per_expert for _ in range(nexperts)] for _ in range(TP_SIZE)], dtype=np.int32)


def generate_ntokens_per_rank_per_expert_random(nexperts_per_rank, nexperts, TP_SIZE):
    return np.array([np.random.multinomial(nexperts_per_rank, [1 / nexperts] * nexperts) for _ in range(TP_SIZE)],
                    dtype=np.int32)


def generate_ntokens_per_rank_per_expert_random_with_many_zeros(nexperts_per_rank, nexperts, TP_SIZE, zero_rate=0.3):

    def _rand():
        if random.random() > 1 - zero_rate:
            return 0
        return random.random()

    weight = np.array([_rand() + 1e-5 for _ in range(nexperts)])
    weight = weight / weight.sum()

    return np.array([np.random.multinomial(nexperts_per_rank, weight) for _ in range(TP_SIZE)])


def _check_tiled_m(tiled_m_global):
    tiled_m_sorted, _ = tiled_m_global.sort()
    torch.testing.assert_close(tiled_m_sorted,
                               torch.arange(0, tiled_m_global.shape[0], device="cuda", dtype=tiled_m_global.dtype))


def check_with_ntokens_per_rank_per_expert(ntokens_per_rank_per_expert: np.ndarray, nexperts, TP_SIZE, LOCAL_TP_SIZE,
                                           BLOCK_SIZE_M, verbose=True):
    for rank in range(TP_SIZE):
        expert_id, tile_index = threadblock_swizzle_ag_moe_triton(
            rank,
            nexperts,
            TP_SIZE,
            LOCAL_TP_SIZE,
            BLOCK_SIZE_M,
            ntokens_per_rank_per_expert,
            verbose=verbose,
        )

        try:
            _check_tiled_m(tile_index)
        except Exception as e:
            logging.fatal(
                f"rank: {rank}, expert_id: {expert_id}, tiled_m_global: {tile_index}, ntokens_per_rank_per_expert: {ntokens_per_rank_per_expert}"
            )
            raise e


def get_default_profile_context():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    )


def test_threadblock_swizzle_ag_moe_naive(nexperts=2, TP_SIZE=4, LOCAL_TP_SIZE=4, BLOCK_SIZE_M=128, verbose=False):
    for ntokens in [
            generate_ntokens_per_rank_per_expert_uniform(BLOCK_SIZE_M * nexperts, nexperts, TP_SIZE),
            generate_ntokens_per_rank_per_expert_uniform((BLOCK_SIZE_M - 1) * nexperts, nexperts, TP_SIZE),
            generate_ntokens_per_rank_per_expert_uniform((BLOCK_SIZE_M + 1) * nexperts, nexperts, TP_SIZE),
    ]:
        check_with_ntokens_per_rank_per_expert(ntokens, nexperts, TP_SIZE, LOCAL_TP_SIZE, BLOCK_SIZE_M, verbose)


def test_threadblock_swizzle_ag_moe_random(nexperts=32, TP_SIZE=8, LOCAL_TP_SIZE=8, BLOCK_SIZE_M=128, verbose=False):
    for n in range(100):
        for _ in range(1000):
            ntokens = generate_ntokens_per_rank_per_expert_random(BLOCK_SIZE_M * nexperts, nexperts, TP_SIZE)
            check_with_ntokens_per_rank_per_expert(ntokens, nexperts, TP_SIZE, LOCAL_TP_SIZE, BLOCK_SIZE_M,
                                                   verbose=verbose)
        print(f"[{n}] random passed...")
        for _ in range(1000):
            ntokens = generate_ntokens_per_rank_per_expert_random_with_many_zeros(BLOCK_SIZE_M * nexperts, nexperts,
                                                                                  TP_SIZE, 0.3)
            check_with_ntokens_per_rank_per_expert(ntokens, nexperts, TP_SIZE, LOCAL_TP_SIZE, BLOCK_SIZE_M,
                                                   verbose=verbose)
        print(f"[{n}] random with many zeroes passed...")


def perf_threadblock_swizzle_ag_moe_random(nexperts=32, TP_SIZE=8, LOCAL_TP_SIZE=8, BLOCK_SIZE_M=128, verbose=False):
    ctx = get_default_profile_context()
    with ctx:
        for n in range(10):
            ntokens = generate_ntokens_per_rank_per_expert_random(BLOCK_SIZE_M * nexperts, nexperts, TP_SIZE)
            check_with_ntokens_per_rank_per_expert(ntokens, nexperts, TP_SIZE, LOCAL_TP_SIZE, BLOCK_SIZE_M,
                                                   verbose=verbose)
    ctx.export_chrome_trace("threadblock_swizzle_ag_moe_triton.json.tar.gz")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf_only", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--naive_only", action="store_true", default=False)
    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--tp_size", "--tp-size", default=8, type=int)
    parser.add_argument("--local_tp_size", "--local-tp-size", default=8, type=int)
    parser.add_argument("--nexperts", default=64, type=int)
    parser.add_argument("--block_size_m", default=128, type=int)
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def _load(filename):
    try:
        import torch
        x = torch.load(filename).cpu().numpy()
        return x
    except Exception:
        x = np.load(filename)
        return x


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    logging.basicConfig(level=logging.INFO)
    import sys

    args = _parse_args()
    assert args.tp_size % args.local_tp_size == 0 and args.tp_size > 0
    if args.from_file:
        ntokens_per_rank_per_expert: np.ndarray = _load(args.from_file)
        print("ntokens_per_rank_per_expert(in)", ntokens_per_rank_per_expert)
        tp_size, nexperts = ntokens_per_rank_per_expert.shape
        if tp_size != args.tp_size:
            print(f"warning: tp_size not matching. use tp_size from file: {tp_size}")
        if nexperts != args.nexperts:
            print(f"warning: nexperts not matching. use nexperts from file: {nexperts}")
        check_with_ntokens_per_rank_per_expert(ntokens_per_rank_per_expert, nexperts, tp_size, args.local_tp_size,
                                               args.block_size_m)
        sys.exit(0)

    if args.perf_only:
        perf_threadblock_swizzle_ag_moe_random(args.nexperts, args.tp_size, args.local_tp_size, args.block_size_m,
                                               args.verbose)
        sys.exit(0)

    test_threadblock_swizzle_ag_moe_naive(args.nexperts, args.tp_size, args.local_tp_size, args.block_size_m,
                                          args.verbose)
    if args.naive_only:
        sys.exit(0)

    #  this takes a lot of time.
    test_threadblock_swizzle_ag_moe_random(args.nexperts, args.tp_size, args.local_tp_size, args.block_size_m,
                                           args.verbose)
