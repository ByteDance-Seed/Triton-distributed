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

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Representative DSL kernels for the ``cuda-compile`` suite.

These functions live in a real file on disk on purpose: the compiler reads
kernel source via :func:`inspect.getsource`, which cannot see ``exec``-ed code.
They cover the four representative families required by RF13 -- scalar,
vector-width, reduction (warp shuffle) and an MMA atom -- and are compiled to
PTX with the available NVCC. No GPU kernel is launched and the result is a
build/toolchain result only, never a numerical or performance claim.
"""

from __future__ import annotations

import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def scalar_vec_add(
    a: ll.Tensor[ll.float32],
    b: ll.Tensor[ll.float32],
    c: ll.Tensor[ll.float32],
    n: ll.int32,
) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        c[i] = a[i] + b[i]


@lk.ll_kernel(backend="cuda", is_entry=True)
def vector_v4_roundtrip(src: ll.Tensor[ll.uint32], dst: ll.Tensor[ll.uint32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    off: ll.int32 = t * 4
    frag = ll.empty([4], dtype=ll.uint32, scope="local")
    ll.ld_global_v4_u32(src, off, frag)
    ll.st_global_v4_u32(dst, off, frag)


@lk.ll_kernel(backend="cuda", is_entry=True)
def warp_shuffle_reduce(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = x[tid]
    v = v + ll.shfl_down_sync_width(0xFFFFFFFF, v, 16, 32)
    v = v + ll.shfl_down_sync_width(0xFFFFFFFF, v, 8, 32)
    v = v + ll.shfl_down_sync_width(0xFFFFFFFF, v, 4, 32)
    v = v + ll.shfl_down_sync_width(0xFFFFFFFF, v, 2, 32)
    v = v + ll.shfl_down_sync_width(0xFFFFFFFF, v, 1, 32)
    out[tid] = v


@lk.ll_kernel(backend="cuda", is_entry=True)
def f64_mma_atom(a: ll.Tensor[ll.float64], b: ll.Tensor[ll.float64], c: ll.Tensor[ll.float64]) -> ll.void:
    acc = ll.empty([4], dtype=ll.float64, scope="local")
    frag_a = ll.empty([2], dtype=ll.float64, scope="local")
    frag_b = ll.empty([1], dtype=ll.float64, scope="local")
    for i in ll.unroll(range(4)):
        acc[i] = 0.0
    for i in ll.unroll(range(2)):
        frag_a[i] = a[i]
    frag_b[0] = b[0]
    ll.mma_m16n8k4_f64(acc, frag_a, frag_b, acc, dtype="f64")
    for i in ll.unroll(range(4)):
        c[i] = acc[i]


# case_id -> (kernel object, threads, (required source markers), nvcc arch)
COMPILE_PROBES = {
    "compile-scalar": (scalar_vec_add, 128, ("+", ), "sm_80"),
    "compile-vector": (vector_v4_roundtrip, 32, ("ld.global.v4.u32", "st.global.v4.u32"), "sm_80"),
    "compile-reduction": (warp_shuffle_reduce, 32, ("shfl", ), "sm_80"),
    # The documented FP64 MMA opcode needs sm_89/sm_90; cross-compile for sm_90a.
    "compile-mma-atom": (f64_mma_atom, 32, ("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64", ), "sm_90a"),
}
