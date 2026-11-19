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

M = 128
N = 128
K = 64
STAGES = 2
NUM_THREADS = 224

SMEM_STRIDE_BYTES = 8 * K * 2


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def empty_umma_sm100_kernel(
    a_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    b_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    out: ll.Tensor[ll.float32],
    num_iters: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_a = g.role("load_a", hw=HW.TMA, warps=[5])
        load_b = g.role("load_b", hw=HW.TMA, warps=[6])
        mma = g.role("mma", hw=HW.TensorCore, warps=[4])
        consume = g.role("consume", hw=HW.CUDACore, warps=[0, 1, 2, 3])

        a_smem = g.smem("a_smem", (M, K), dtype=ll.bfloat16, stages=STAGES, major="K")
        b_smem = g.smem("b_smem", (M, K), dtype=ll.bfloat16, stages=STAGES, major="K")
        accum = g.tmem("accum", (M, N), dtype=ll.float32, stages=STAGES)

        input_stage = g.depends(src=[load_a, load_b], dst=mma, via=[a_smem, b_smem])
        output_stage = g.depends(src=mma, dst=consume, via=[accum])

    tid: ll.int32 = ll.threadIdx_x()

    with load_a:
        issuer_a: ll.uint32 = ll.elect_one()
        i_a: ll.int32 = 0
        while i_a < num_iters:
            input_stage.bind(i_a)
            if issuer_a == 1:
                ri.tma_ld_2d(a_smem, a_desc, (0, 0), shape=(M, K))
            i_a = i_a + 1

    with load_b:
        issuer_b: ll.uint32 = ll.elect_one()
        i_b: ll.int32 = 0
        while i_b < num_iters:
            input_stage.bind(i_b)
            if issuer_b == 1:
                ri.tma_ld_2d(b_smem, b_desc, (0, 0), shape=(M, K))
            i_b = i_b + 1

    with mma:
        issuer_mma: ll.uint32 = ll.elect_one()
        i_mma: ll.int32 = 0
        while i_mma < num_iters:
            input_stage.bind(i_mma)
            output_stage.bind(i_mma)
            if i_mma % 4 != 1:
                if issuer_mma == 1:
                    ad: ll.uint64 = ri.make_umma_smem_desc(a_smem, (0, 0))
                    bd: ll.uint64 = ri.make_umma_smem_desc(b_smem, (0, 0))
                    ri.umma_f16_ss_1sm_kblock(
                        accum,
                        ad,
                        bd,
                        K,
                        out_shape=(M, N),
                        zero_init=True,
                    )
            i_mma = i_mma + 1

    with consume:
        i_consume: ll.int32 = 0
        while i_consume < num_iters:
            output_stage.bind(i_consume)
            if i_consume % 4 != 1:
                r = ll.empty([4], dtype=ll.uint32, scope="local")
                ri.tmem_ld_32dp_32bxn(accum, ll.uint32(accum), r, 4)
                ri.tmem_ld_fence()
                out[i_consume * M + tid] = ll.uint_as_float(r[0])
            else:
                out[i_consume * M + tid] = -1.0
            i_consume = i_consume + 1
