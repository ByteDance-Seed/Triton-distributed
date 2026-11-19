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
def empty_tx_sm100_kernel(
    src_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    out: ll.Tensor[ll.float32],
    num_iters: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        consume = g.role("consume", hw=HW.CUDACore, warps=[1])
        tile = g.smem("tile", (1, TILE), dtype=ll.float32, stages=STAGES)
        stage = g.depends(src=load, dst=consume, via=[tile])

    lane: ll.int32 = ll.get_lane_idx()

    with load:
        issuer: ll.uint32 = ll.elect_one()
        i_load: ll.int32 = 0
        while i_load < num_iters:
            stage.bind(i_load)
            if i_load % 4 != 1:
                if issuer == 1:
                    ri.tma_ld_2d(tile, src_desc, (0, i_load), shape=(1, TILE))
            i_load = i_load + 1

    with consume:
        i_consume: ll.int32 = 0
        while i_consume < num_iters:
            stage.bind(i_consume)
            if i_consume % 4 != 1:
                out[i_consume * TILE + lane] = ri.smem_ld_f32(tile, lane)
            else:
                out[i_consume * TILE + lane] = -1.0
            i_consume = i_consume + 1
