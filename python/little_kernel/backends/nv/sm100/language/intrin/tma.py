"""SM100 two-CTA TMA instructions and peer-barrier encoding."""
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

from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.type_system import void
from little_kernel.backends.nv.common.language.intrin.tma import _codegen_tma_load_mcast


def codegen_tma_load_3d_2sm(desc, bar_ptr, smem_ptr, c0, c1, c2):
    body = """
__device__ __forceinline__ void tma_load_3d_2sm_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(ba) : "memory");
}
"""
    return Builtin(
        body=body,
        includes=["<cuda.h>"],
        return_val=f"tma_load_3d_2sm_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1}, {c2})",
    )


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_3d_2sm)
def tma_load_3d_2sm(desc, bar_ptr, smem_ptr, c0, c1, c2):
    """cta_group::2 TMA 3D load. Applies PEER_BIT_MASK to barrier address."""
    raise RuntimeError("should not call tma_load_3d_2sm in compilation")


def codegen_tma_load_2d_2sm(desc, bar_ptr, smem_ptr, c0, c1):
    body = """
__device__ __forceinline__ void tma_load_2d_2sm_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_load_2d_2sm_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_2d_2sm)
def tma_load_2d_2sm(desc, bar_ptr, smem_ptr, c0, c1):
    """cta_group::2 TMA 2D load. Applies PEER_BIT_MASK to barrier address."""
    raise RuntimeError("should not call tma_load_2d_2sm in compilation")


def codegen_tma_load_2d_2sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask, issue_pred=1):
    return _codegen_tma_load_mcast(2, 2, desc, bar_ptr, smem_ptr, (c0, c1), multicast_mask, issue_pred,
                                   peer_mask=" & 0xFEFFFFFF", group_qualifier=".cta_group::2")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_2d_2sm_mcast)
def tma_load_2d_2sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask, issue_pred=1):
    raise RuntimeError("should not call tma_load_2d_2sm_mcast in compilation")


def codegen_tma_load_3d_2sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, multicast_mask, issue_pred=1):
    return _codegen_tma_load_mcast(3, 2, desc, bar_ptr, smem_ptr, (c0, c1, c2), multicast_mask, issue_pred,
                                   peer_mask=" & 0xFEFFFFFF", group_qualifier=".cta_group::2")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_3d_2sm_mcast)
def tma_load_3d_2sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, multicast_mask, issue_pred=1):
    raise RuntimeError("should not call tma_load_3d_2sm_mcast in compilation")
