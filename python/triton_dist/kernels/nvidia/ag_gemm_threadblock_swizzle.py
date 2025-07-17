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
import triton
import triton.language as tl
from triton.language.extra.cuda.language_extra import (
    laneid,
    __shfl_up_sync_i32,
    __shfl_down_sync_i32,
    __shfl_sync_i32,
    ffs,
    __ballot_sync,
    st,
    tid,
)


@triton.jit
def warp_prefix_sum_kernel(value, lane_id, len):
    i = 1
    while i < min(len * 2, 32):
        val = __shfl_up_sync_i32(0xFFFFFFFF, value, i)
        if lane_id >= i:
            value += val
        i = i * 2

    return value


@triton.jit(do_not_specialize=["rank"])
def threadblock_swizzle_allgather_gemm_kernel(
    tiled_m,
    M,
    rank,
    WORLD_SIZE: tl.constexpr,
    NNODES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    DEBUG: tl.constexpr = False,
):
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    node_id = rank // LOCAL_WORLD_SIZE
    M_per_rank = M // WORLD_SIZE
    M_per_node = M // NNODES
    node_start = node_id

    lane_id = laneid()

    if lane_id < NNODES:
        n = (lane_id + node_start) % NNODES
        M_node_start = M_per_node * n
        M_node_end = M_per_node * (n + 1)
        tiled_m_node_start = M_node_start // BLOCK_SIZE_M
        prev_tiled_m_node_end = (M_node_start - 1) // BLOCK_SIZE_M
        tiled_m_node_end = (M_node_end - 1) // BLOCK_SIZE_M
        next_tiled_m_node_start = M_node_end // BLOCK_SIZE_M

        if lane_id == 0 and M_node_start != 0:
            if prev_tiled_m_node_end == tiled_m_node_start:
                tiled_m_node_start += 1

        if lane_id == 0 and M_node_end != M:
            if next_tiled_m_node_start == tiled_m_node_end:
                tiled_m_node_end -= 1

        if lane_id != NNODES - 1 and M_node_end != M:
            if next_tiled_m_node_start == tiled_m_node_end:
                tiled_m_node_end -= 1
        if DEBUG and lane_id == NNODES - 1:
            print("tiled_m_node_end", tiled_m_node_end)

        swizzled_tiled_m_size = tiled_m_node_end - tiled_m_node_start + 1
    else:
        swizzled_tiled_m_size = 0

    if DEBUG and lane_id < NNODES:
        print("swizzled_tiled_m_size", swizzled_tiled_m_size, lane_id)
    swizzled_tiled_m_size_accum = (warp_prefix_sum_kernel(swizzled_tiled_m_size, lane_id, NNODES) -
                                   swizzled_tiled_m_size)
    if DEBUG and lane_id < NNODES:
        print("swizzled_tiled_m_size_accum", swizzled_tiled_m_size_accum)

    tiled_m_size_l = __shfl_down_sync_i32(0xFFFFFFFF, swizzled_tiled_m_size, NNODES - node_start)
    tiled_m_size_r = __shfl_up_sync_i32(0xFFFFFFFF, swizzled_tiled_m_size, node_start)
    tiled_m_size = 0
    if lane_id < node_start:
        tiled_m_size = tiled_m_size_l
    elif lane_id < NNODES:
        tiled_m_size = tiled_m_size_r

    if DEBUG and lane_id < NNODES:
        print("tiled_m_size", tiled_m_size)

    tiled_m_size_accum = warp_prefix_sum_kernel(tiled_m_size, lane_id, NNODES) - tiled_m_size
    mask = __ballot_sync(0xFFFFFFFF, tiled_m < swizzled_tiled_m_size_accum)
    n = ffs(mask) - 1 - 1
    if DEBUG and lane_id < NNODES:
        print("tiled_m_size_accum", tiled_m_size_accum)
        print("n", n, tiled_m, swizzled_tiled_m_size_accum, mask)

    # map node
    nid = (n + node_start) % NNODES
    node_offset = __shfl_sync_i32(0xFFFFFFFF, swizzled_tiled_m_size_accum, n)

    tile_size = __shfl_sync_i32(0xFFFFFFFF, swizzled_tiled_m_size, n)

    tiled_m_intra_node = tiled_m - node_offset
    local_rank = rank % LOCAL_WORLD_SIZE
    m_start = M_per_node * nid + M_per_rank * local_rank
    tiled_m_start = tl.cdiv(m_start, BLOCK_SIZE_M)
    swizzled_node_offset = __shfl_sync_i32(0xFFFFFFFF, tiled_m_size_accum, nid)
    rank_offset = max(0, tiled_m_start - swizzled_node_offset)  # this may < 0, bad

    # map rank
    tiled_m_intra_node_new = (tiled_m_intra_node + rank_offset) % tile_size
    return swizzled_node_offset + tiled_m_intra_node_new


# naive swizzle algorithm only for perfect shape
@triton.jit
def naive_thread_block_swizzle_allgather_gemm_kernel(
    tiled_m,
    M,
    rank,
    WORLD_SIZE: tl.constexpr,
    NNODES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
    node_id = rank // LOCAL_WORLD_SIZE
    M_per_rank = M // WORLD_SIZE
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    m_rank = tiled_m // pid_ms_per_rank
    pid_m_intra_rank = tiled_m - m_rank * pid_ms_per_rank
    m_node_id = m_rank // LOCAL_WORLD_SIZE
    m_local_rank = m_rank % LOCAL_WORLD_SIZE
    swizzle_m_node_id = (m_node_id + node_id) % NNODES
    swizzle_m_local_rank = (m_local_rank + rank) % LOCAL_WORLD_SIZE
    swizzle_m_rank = swizzle_m_node_id * LOCAL_WORLD_SIZE + swizzle_m_local_rank

    return swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank


if __name__ == "__main__":
    from functools import partial
    import logging
    import numpy as np
    import torch
    from triton_dist.utils import perf_func

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=999)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def cdiv(n, m):
        return (n - 1 + m) // m

    # Just for verification
    def threadblock_swizzle_allgather_gemm_torch(tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M):
        LOCAL_WORLD_SIZE = WORLD_SIZE // NNODES
        node_id = rank // LOCAL_WORLD_SIZE
        M_per_rank = M // WORLD_SIZE
        M_per_node = M // NNODES
        node_start = node_id

        swizzled_tiled_m_sizes = np.empty(NNODES, dtype=np.int32)
        for i in range(NNODES):
            n = (i + node_start) % NNODES
            M_node_start = M_per_node * n
            M_node_end = M_per_node * (n + 1)
            tiled_m_node_start = M_node_start // BLOCK_SIZE_M
            tiled_m_node_end = (M_node_end - 1) // BLOCK_SIZE_M
            prev_tiled_m_node_end = (M_node_start - 1) // BLOCK_SIZE_M
            next_tiled_m_node_start = M_node_end // BLOCK_SIZE_M
            if tiled_m == 0:
                logging.debug(
                    f"rank {rank}, node_id {node_start}, n {n} == M_node_start {M_node_start}, M_node_end {M_node_end}")
                logging.debug(
                    f"rank {rank}, node_id {node_start}, n {n} == tiled_m_node_start {tiled_m_node_start}, tiled_m_node_end {tiled_m_node_end}"
                )
                logging.debug(
                    f"rank {rank}, node_id {node_start}, n {n} == prev_tiled_m_node_end {prev_tiled_m_node_end}, next_tiled_m_node_start {next_tiled_m_node_start}"
                )

            if i == 0 and M_node_start != 0:
                if prev_tiled_m_node_end == tiled_m_node_start:
                    tiled_m_node_start += 1

            if i == 0 and M_node_end != M:
                if next_tiled_m_node_start == tiled_m_node_end:
                    tiled_m_node_end -= 1

            if i != NNODES - 1 and M_node_end != M:
                if next_tiled_m_node_start == tiled_m_node_end:
                    tiled_m_node_end -= 1

            if tiled_m == 0:
                logging.debug(
                    f"rank {rank}, node_id {node_start}, n {n} => tiled_m_node_start {tiled_m_node_start}, tiled_m_node_end {tiled_m_node_end}"
                )

            swizzled_tiled_m_sizes[i] = tiled_m_node_end - tiled_m_node_start + 1

        swizzled_tiled_m_sizes_accum = np.cumsum(swizzled_tiled_m_sizes)
        swizzled_tiled_m_sizes_accum = np.insert(swizzled_tiled_m_sizes_accum, 0, 0)

        tiled_m_sizes = np.concatenate((swizzled_tiled_m_sizes[-node_start:], swizzled_tiled_m_sizes[:-node_start]))
        tiled_m_sizes_accum = np.cumsum(tiled_m_sizes)
        tiled_m_sizes_accum = np.insert(tiled_m_sizes_accum, 0, 0)

        if tiled_m == 0:
            logging.debug(
                f"swizzled_tiled_m_sizes {swizzled_tiled_m_sizes}, tiled_m_sizes {tiled_m_sizes}, tiled_m_sizes_accum: {tiled_m_sizes_accum}"
            )

        # upper bound
        for n in range(NNODES + 1):
            if tiled_m < swizzled_tiled_m_sizes_accum[n]:
                break

        n = n - 1

        # map node
        nid = (n + node_start) % NNODES
        node_offset = swizzled_tiled_m_sizes_accum[n]

        tiled_m_intra_node = tiled_m - node_offset
        local_rank = rank % LOCAL_WORLD_SIZE
        m_start = M_per_node * nid + M_per_rank * local_rank
        tiled_m_start = cdiv(m_start, BLOCK_SIZE_M)
        swizzled_node_offset = tiled_m_sizes_accum[nid]
        rank_offset = max(0, tiled_m_start - swizzled_node_offset)  # this may < 0, bad
        logging.debug(
            f"tiled_m: {tiled_m} @ node {n}: tiled_m_start = {tiled_m_start}, swizzled_node_offset = {swizzled_node_offset}, rank_offset = {rank_offset}"
        )

        # map rank
        tiled_m_intra_node_new = (tiled_m_intra_node + rank_offset) % swizzled_tiled_m_sizes[n]

        logging.debug(
            f"tiled_m: {tiled_m} @ node {n}: nid = {nid}, node_offset = {node_offset}, rank_offset = {rank_offset}, m_start={m_start}, tiled_m_sizes_accum[{n}], tiled_m_intra_node = {tiled_m_intra_node}, swizzled_offset = {swizzled_node_offset} + {tiled_m_intra_node_new}"
        )

        return swizzled_node_offset + tiled_m_intra_node_new

    @triton.jit
    def _threadblock_complex_swizzle_run(
        output,
        tiled_m,
        M,
        rank,
        WORLD_SIZE: tl.constexpr,
        NNODES: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        DEBUG: tl.constexpr = False,
    ):
        thread_idx = tid(0)
        tiled_m_new = threadblock_swizzle_allgather_gemm_kernel(tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M,
                                                                DEBUG)
        if thread_idx == 0:
            st(output, tiled_m_new)

    @triton.jit
    def _threadblock_naive_swizzle_run(output, tiled_m, M, rank, WORLD_SIZE: tl.constexpr, NNODES: tl.constexpr,
                                       BLOCK_SIZE_M: tl.constexpr):
        thread_idx = tid(0)
        tiled_m_new = naive_thread_block_swizzle_allgather_gemm_kernel(tiled_m, M, rank, WORLD_SIZE, NNODES,
                                                                       BLOCK_SIZE_M)
        if thread_idx == 0:
            st(output, tiled_m_new)

    # use 1 sm to calculate the sizzle results of given tile_id
    def threadblock_complex_swizzle_allgather_gemm_triton(tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M):
        output = torch.empty((1, ), dtype=torch.int32, device="cuda")
        _threadblock_complex_swizzle_run[(1, )](output, tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M, num_warps=1,
                                                DEBUG=args.debug)
        return int(output.item())

    def threadblock_naive_swizzle_allgather_gemm_triton(tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M):
        output = torch.empty((1, ), dtype=torch.int32, device="cuda")
        _threadblock_naive_swizzle_run[(1, )](output, tiled_m, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M, num_warps=1)
        return int(output.item())

    LOCAL_WORLD_SIZE = 8
    NNODES = 2
    WORLD_SIZE = LOCAL_WORLD_SIZE * NNODES
    BLOCK_SIZE_M = 128
    M = args.M * WORLD_SIZE
    N_per_rank = args.N
    K = args.K

    def _check_threadblock_swizzle_with(M):
        logging.info(
            f"M: {M}, M_per_rank: {M // WORLD_SIZE}, M_per_node: {M // NNODES}, BLOCK_SIZE_M: {BLOCK_SIZE_M}, BLOCK_NUM: {cdiv(M, BLOCK_SIZE_M)}"
        )

        for rank in range(WORLD_SIZE):
            swizzled = []
            for n in range(cdiv(M, BLOCK_SIZE_M)):
                x = threadblock_complex_swizzle_allgather_gemm_triton(n, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
                y = threadblock_swizzle_allgather_gemm_torch(n, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
                if M % BLOCK_SIZE_M == 0:  # test results of naive swizzle under perfect shape
                    z = threadblock_naive_swizzle_allgather_gemm_triton(n, M, rank, WORLD_SIZE, NNODES, BLOCK_SIZE_M)
                    assert x == z, f"{n} => {x} {z}"
                assert x == y, f"{n} => {x} {y}"
                assert x < cdiv(M, BLOCK_SIZE_M), f"{n} => {x}"
                swizzled.append(x)

            logging.info(f"complex swizzle {rank:02}: {swizzled}")
            swizzled.sort()
            for i in range(len(swizzled)):
                if swizzled[i] != i:
                    raise Exception(f"swizzled: {swizzled}")

    def _bench_threadblock_swizzle():
        complex_swizzle_perf = []
        for rank in range(WORLD_SIZE):
            _, perf = perf_func(
                partial(threadblock_complex_swizzle_allgather_gemm_triton, 1, M, rank, WORLD_SIZE, NNODES,
                        BLOCK_SIZE_M), iters=100, warmup_iters=500)
            complex_swizzle_perf.append(perf)
        logging.info(f"swizzle perf => {complex_swizzle_perf} ms")

        if M % BLOCK_SIZE_M == 0:
            naive_swizzle_perf = []
            for rank in range(WORLD_SIZE):
                _, perf = perf_func(
                    partial(threadblock_naive_swizzle_allgather_gemm_triton, 1, M, rank, WORLD_SIZE, NNODES,
                            BLOCK_SIZE_M), iters=100, warmup_iters=500)
                naive_swizzle_perf.append(perf)
            logging.info(f"naive swizzle perf => {naive_swizzle_perf} ms")

    _check_threadblock_swizzle_with(M)
    _bench_threadblock_swizzle()
