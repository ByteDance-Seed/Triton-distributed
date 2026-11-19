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

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

TILE = 32
STAGES = 2
NUM_THREADS = 64


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def empty_tma_store_sm100_kernel(
    out_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    num_iters: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        produce = g.role("produce", hw=HW.CUDACore, warps=[0])
        store = g.role("store", hw=HW.TMA, warps=[1])
        tile = g.smem("tile", (1, TILE), dtype=ll.float32, stages=STAGES)
        stage = g.depends(src=produce, dst=store, via=[tile])

    lane: ll.int32 = ll.get_lane_idx()

    with produce:
        i_produce: ll.int32 = 0
        while i_produce < num_iters:
            stage.bind(i_produce)
            ri.smem_st_f32(tile, lane, ll.float32(i_produce * TILE + lane))
            i_produce = i_produce + 1

    with store:
        i_store: ll.int32 = 0
        while i_store < num_iters:
            issuer: ll.bool_ = lane == i_store % 32
            stage.bind(i_store)
            if i_store % 4 != 1:
                if issuer:
                    ri.tma_st_2d(tile, out_desc, (0, i_store), shape=(1, TILE))
            i_store = i_store + 1
