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
import os

import torch
import torch.distributed

import triton_dist
import triton.language as tl
from triton_dist.language.extra.cuda.language_extra import (tid)
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (init_nvshmem_by_torch_process_group, finalize_distributed, nvshmem_barrier_all_on_stream,
                               nvshmem_free_tensor_sync, nvshmem_create_tensor)
import datetime


def test_nvshmemx_putmem_with_scope(pg, N, dtype: torch.dtype = torch.int8):

    @triton_dist.jit
    def _nvshmemx_putmem(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_nbi(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_putmem_block",
                ("warp", False): "nvshmemx_putmem_warp",
                ("thread", False): "nvshmem_putmem",
                ("block", True): "nvshmemx_putmem_nbi_block",
                ("warp", True): "nvshmemx_putmem_nbi_warp",
                ("thread", True): "nvshmem_putmem_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(pg.rank() + 1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemx_putmem[(pg.size(), )](
                t,
                N // pg.size(),
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1,
                                       pg.size() + 1, dtype=dtype, device="cuda").reshape(
                                           (pg.size(), 1)).repeat(1, N // pg.size()).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(pg.size(), -1))
                raise (e)
            else:
                print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(t)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group

    pg1 = torch.distributed.new_group(ranks=list(range(0, WORLD_SIZE // 2)), backend="nccl")
    pg2 = torch.distributed.new_group(ranks=list(range(WORLD_SIZE // 2, WORLD_SIZE)), backend="nccl")

    if RANK < WORLD_SIZE // 2:
        cur_pg = pg1
    else:
        cur_pg = pg2

    init_nvshmem_by_torch_process_group(cur_pg)
    print(f"Rank {RANK} Initialization Done!")
    test_nvshmemx_putmem_with_scope(cur_pg, 512)
    finalize_distributed()
