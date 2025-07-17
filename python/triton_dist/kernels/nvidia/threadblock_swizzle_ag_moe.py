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

import bisect
import dataclasses
import logging
import random
from typing import List, Tuple
import numpy as np


def cdiv(x, y):
    return (x - 1 + y) // y


@dataclasses.dataclass
class Tile:
    expert_id: int
    tiled_m: int
    segment_start: int
    segment_end: int


def cumsum(x):
    y = []
    s = 0
    for n in x:
        s += n
        y.append(s)
    return y


def _split_tiles_for_each_segment(
    expert_id,
    rank,
    tp_size,
    block_size_m,
    token_cnts: List[int],
) -> List[List[Tile]]:
    tiles = []
    token_cnts_acc = cumsum(token_cnts)
    # start from tokens_cnts[rank]

    tiles = [[] for _ in range(tp_size)]

    ntokens = sum(token_cnts)

    global_m_start = token_cnts_acc[rank] - token_cnts[rank]
    # start from here
    global_tiled_m_start = cdiv(global_m_start, block_size_m)
    n_tiles = cdiv(ntokens, block_size_m)
    DBG(f"ntiles: {n_tiles} with {token_cnts_acc} with global_tiled_m_start: {global_m_start}/{global_tiled_m_start} @ {rank}"
        )
    stage_old = -1
    # calculate rank first
    for tid in list(range(global_tiled_m_start, n_tiles)) + list(range(0, global_tiled_m_start)):
        m_start = tid * block_size_m
        m_end = min((tid + 1) * block_size_m, ntokens) - 1
        segment_start = bisect.bisect_right(token_cnts_acc, m_start)
        segment_end = bisect.bisect_right(token_cnts_acc, m_end)
        # the problem is, m_semgent_end may overlap with start, and m_segment_end may even larger than
        stage = (segment_end - rank + tp_size) % tp_size
        # take care for the last tile: may overlap with the first one
        if tid == global_tiled_m_start - 1:
            # if has overlap with the start tile
            if global_m_start % BLOCK_SIZE_M != 0:
                m_segment_end_exclude_first_segment = bisect.bisect_right(token_cnts_acc[:rank], m_end)
                m_segment_end_exclude_first_segment = (m_segment_end_exclude_first_segment -
                                                       1 if m_segment_end_exclude_first_segment == rank else
                                                       m_segment_end_exclude_first_segment)
                DBG(f"m_segment_end: {segment_end} => {m_segment_end_exclude_first_segment}")
                stage = (m_segment_end_exclude_first_segment - rank + tp_size) % tp_size
                assert stage >= stage_old
                stage_old = stage

        DBG(f"{tid} @ stage {stage} m_start: ({m_start}, {m_end}) => ({segment_start}, {segment_end})")
        tiles[stage].append(
            Tile(
                expert_id=expert_id,
                tiled_m=tid,
                segment_start=segment_start,
                segment_end=segment_end,
            ))

    return tiles


def reshape_2d(arr_2d: List[List[Tile]], ) -> List[Tile]:
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


def threadblock_swizzle_ag_moe(tiled_m, rank, nexperts: int, tp_size: int, block_size_m: int,
                               token_cnts_per_rank_per_expert: List[List[int]],  # of shape [nexperts, tp_size]
                               ):
    token_cnts_per_expert_per_rank = reshape_2d(token_cnts_per_rank_per_expert)
    tiles_by_expert_by_segment = [
        _split_tiles_for_each_segment(expert_id, rank, tp_size, block_size_m, token_cnts)
        for expert_id, token_cnts in enumerate(token_cnts_per_expert_per_rank)
    ]
    DBG("tiles_by_expert_by_segment")
    DBG(tiles_by_expert_by_segment)
    tiles_by_segment_by_expert = reshape_2d(tiles_by_expert_by_segment)
    ntiles_by_segment_by_expert = [[len(x) for x in y] for y in tiles_by_segment_by_expert]
    ntiles_acc_by_segment_by_expert = [cumsum(x) for x in ntiles_by_segment_by_expert]
    DBG("tiles_by_segment_by_expert")
    DBG(tiles_by_segment_by_expert)
    ntiles_by_segment = [sum(x) for x in ntiles_by_segment_by_expert]
    DBG(f"ntiles_by_segment: {ntiles_by_segment}")
    ntiles_acc_by_segment = cumsum(ntiles_by_segment)
    DBG(f"ntiles_acc_by_segment: {ntiles_acc_by_segment}")

    # find which segment which expert tiled_m belongs to
    stage = bisect.bisect_right(ntiles_acc_by_segment, tiled_m)
    DBG(f"tiled_m: {tiled_m}")
    DBG(f"segment: {stage}")

    tiled_m_in_rank = tiled_m - (ntiles_acc_by_segment[stage] - ntiles_by_segment[stage])
    DBG(f"tiled_m_in_rank: {tiled_m_in_rank} with {ntiles_acc_by_segment_by_expert[stage]}")
    expert_id = bisect.bisect_right(ntiles_acc_by_segment_by_expert[stage], tiled_m_in_rank)
    DBG(f"expert_id: {expert_id}")

    tiled_m_in_problem = (tiled_m_in_rank - ntiles_acc_by_segment_by_expert[stage][expert_id])
    # now we have tiled_m => (segment, expert_id, tiled_m_in_problem)
    # all we have to do is construct a gather_a, which is sorted_gather_a
    return (
        stage,
        expert_id,
        tiles_by_segment_by_expert[stage][expert_id][tiled_m_in_problem],
    )


def check_swizzled(swizzled: List[Tuple[int, int]], token_cnts_per_rank_per_expert):
    token_cnts_per_expert_per_rank = reshape_2d(token_cnts_per_rank_per_expert)
    # check each expert all tiles is calculated
    nexperts = len(token_cnts_per_expert_per_rank)
    for expert_id in range(nexperts):
        tokens_this_ep = sum(token_cnts_per_expert_per_rank[expert_id])
        num_tiles_this_ep = cdiv(tokens_this_ep, BLOCK_SIZE_M)
        tiled_m = [tiled_m for (eid, tiled_m) in swizzled if eid == expert_id]
        tiled_m.sort()
        assert tiled_m == list(range(num_tiles_this_ep))


def generate_token_cnts_per_rank_per_expert_uniform(ntokens_per_rank, nexperts, TP_SIZE):
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
    return [[ntokens_per_expert for _ in range(nexperts)] for _ in range(TP_SIZE)]


def generate_token_cnts_per_rank_per_expert_random(nexperts_per_rank, nexperts, TP_SIZE):
    return [np.random.multinomial(nexperts_per_rank, [1 / nexperts] * nexperts) for _ in range(TP_SIZE)]


def generate_token_cnts_per_rank_per_expert_random_with_many_zeros(nexperts_per_rank, nexperts, TP_SIZE, zero_rate=0.3):

    def _rand():
        if random.random() > 1 - zero_rate:
            return 0
        return random.random()

    weight = np.array([_rand() + 1e-5 for _ in range(nexperts)])
    weight = weight / weight.sum()

    return [np.random.multinomial(nexperts_per_rank, weight) for _ in range(TP_SIZE)]


def check_with_token_cnt_per_rank_per_expert(token_cnts_per_rank_per_expert, verbose=True):
    token_cnts_per_expert_per_rank = reshape_2d(token_cnts_per_rank_per_expert)
    ntiles_total = sum(
        [cdiv(sum(token_cnts_per_expert), BLOCK_SIZE_M) for token_cnts_per_expert in token_cnts_per_expert_per_rank])
    for rank in range(TP_SIZE):
        # rank = 1
        swizzled = []
        for tiled_m in range(ntiles_total):
            segment, expert_id, tile = threadblock_swizzle_ag_moe(
                tiled_m,
                rank,
                nexperts,
                TP_SIZE,
                BLOCK_SIZE_M,
                token_cnts_per_rank_per_expert,
            )
            swizzled.append([expert_id, tile.tiled_m])
            DBG("\n")
        if verbose:
            print(swizzled)
        try:
            check_swizzled(swizzled, token_cnts_per_rank_per_expert)
        except Exception as e:
            logging.fatal(
                f"rank: {rank}, swizzled: {swizzled}, token_cnts_per_rank_per_expert: {token_cnts_per_rank_per_expert}")
            raise e

        # break


logging.basicConfig(level=logging.INFO)

DBG = logging.debug
# DBG = pprint

TP_SIZE = 4
nexperts = 2
BLOCK_SIZE_M = 128
for token_cnts in [
        generate_token_cnts_per_rank_per_expert_uniform(BLOCK_SIZE_M * nexperts, nexperts, TP_SIZE),
        generate_token_cnts_per_rank_per_expert_uniform((BLOCK_SIZE_M - 1) * nexperts, nexperts, TP_SIZE),
        generate_token_cnts_per_rank_per_expert_uniform((BLOCK_SIZE_M + 1) * nexperts, nexperts, TP_SIZE),
]:
    check_with_token_cnt_per_rank_per_expert(token_cnts)

# set TP_SIZE=4 and nexperts = 2 is too slow to run for python.
TP_SIZE = 4
nexperts = 2

for n in range(100):
    for n in range(1000):
        token_cnts = generate_token_cnts_per_rank_per_expert_random(BLOCK_SIZE_M * nexperts, nexperts, TP_SIZE)
        check_with_token_cnt_per_rank_per_expert(token_cnts, verbose=False)
    print("[n] random passed...")
    for n in range(1000):
        token_cnts = generate_token_cnts_per_rank_per_expert_random_with_many_zeros(BLOCK_SIZE_M * nexperts, nexperts,
                                                                                    TP_SIZE, 0.3)
        check_with_token_cnt_per_rank_per_expert(token_cnts, verbose=False)
    print("[n] random with many zeroes passed...")
