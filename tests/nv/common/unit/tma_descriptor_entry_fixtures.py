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
"""Fixture kernels for test_tma_descriptor_entry_param_type.

Kernels are defined in a real module (not inline in the test) because the
frontend resolves kernel bodies via ``inspect.getsource``.
"""

import little_kernel as lk
import little_kernel.language as ll

BM = 8
BN = 64


@lk.ll_kernel(backend='cuda', is_entry=True)
def bare_shorthand_entry(dA: ll.TmaDescriptor, out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(dA, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)
    ll.__syncthreads()


@lk.ll_kernel(backend='cuda', is_entry=True)
def const_shorthand_entry(dA: ll.const[ll.TmaDescriptor], out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(dA, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)
    ll.__syncthreads()


@lk.ll_kernel(backend='cuda', is_entry=True)
def explicit_grid_const_entry(dA: ll.const[ll.grid_constant[ll.TmaDescriptor]], out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(dA, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)
    ll.__syncthreads()


@lk.ll_kernel(backend='cuda', is_entry=False)
def tma_load_helper(d: ll.TmaDescriptor, bar: ll.Tensor[ll.int64], tile: ll.Tensor[ll.bfloat16]) -> ll.void:
    ll.mbarrier_arrive_and_expect_tx(bar, BM * BN * 2)
    ll.tma_load_2d_1sm(d, bar, tile, 0, 0)
    ll.mbarrier_wait(bar, 0)


@lk.ll_kernel(backend='cuda', is_entry=True)
def device_caller_entry(dA: ll.TmaDescriptor, out: ll.Tensor[ll.bfloat16]) -> ll.void:
    tile = ll.alloc_shared_memory("s_tile", ll.bfloat16, BM * BN)
    bar = ll.alloc_shared_memory("s_bar", ll.int64, 1)
    ll.init_smem_barrier(bar, 1)
    tma_load_helper(dA, bar, tile)
    ll.__syncthreads()
