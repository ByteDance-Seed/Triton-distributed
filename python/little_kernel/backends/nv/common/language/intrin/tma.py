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
"""
TMA (Tensor Memory Accelerator) intrinsics for SM90+.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import void

# ==============================================================================
# TMA Load
# ==============================================================================


@builtin
class tma_load_2d_1sm(BuiltinOp):
    """Single-SM TMA 2D load from global to shared memory."""

    @staticmethod
    def return_type(desc, bar_ptr, smem_ptr, c0, c1):
        return void

    @staticmethod
    def codegen(desc, bar_ptr, smem_ptr, c0, c1):
        body = """
__device__ __forceinline__ void tma_load_2d_1sm_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "r"(c0), "r"(c1) : "memory");
}
"""
        return Builtin(body=body, includes=["<cuda.h>"],
                       return_val=f"tma_load_2d_1sm_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1})")


# ==============================================================================
# TMA 3D Load (for Flash Attention Q/K/V)
# Uses same descriptor passing pattern as tma_load_2d_1sm: wrapper receives pointer
# ==============================================================================


@builtin
class tma_load_3d_1sm(BuiltinOp):
    """Single-SM TMA 3D load from global to shared memory."""

    @staticmethod
    def return_type(desc, bar_ptr, smem_ptr, c0, c1, c2, cache_hint=None):
        return void

    @staticmethod
    def codegen(desc, bar_ptr, smem_ptr, c0, c1, c2, cache_hint=None):
        cache_hint = 0 if cache_hint is None else cache_hint
        body = """
__device__ __forceinline__ void tma_load_3d_1sm_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1), "r"(c2), "l"(hint) : "memory");
}
"""
        return Builtin(body=body, includes=["<cuda.h>"],
                       return_val=f"tma_load_3d_1sm_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1}, {c2}, {cache_hint})")


# ==============================================================================
# TMA 4D Load (rank-4 tensor maps, e.g. a BSHD attention operand viewed
# innermost-out as (D, S, H, B)).  Uses the same descriptor passing pattern
# as tma_load_2d_1sm: the wrapper receives a pointer to the by-value map.
# ==============================================================================


@builtin
class tma_load_4d_1sm(BuiltinOp):
    """Single-SM TMA 4D load from global to shared memory."""

    @staticmethod
    def return_type(desc, bar_ptr, smem_ptr, c0, c1, c2, c3, cache_hint=None):
        return void

    @staticmethod
    def codegen(desc, bar_ptr, smem_ptr, c0, c1, c2, c3, cache_hint=None):
        cache_hint = 0 if cache_hint is None else cache_hint
        body = """
__device__ __forceinline__ void tma_load_4d_1sm_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3, uint64_t hint) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1), "r"(c2), "r"(c3), "l"(hint) : "memory");
}
"""
        return Builtin(
            body=body, includes=["<cuda.h>"],
            return_val=f"tma_load_4d_1sm_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1}, {c2}, {c3}, {cache_hint})")


# ==============================================================================
# TMA Store
# ==============================================================================


@builtin
class tma_store_2d(BuiltinOp):
    """TMA 2D store from shared to global memory."""

    @staticmethod
    def return_type(desc, smem_ptr, c0, c1):
        return void

    @staticmethod
    def codegen(desc, smem_ptr, c0, c1):
        body = """
__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "r"(c0), "r"(c1) : "memory");
}
"""
        return Builtin(body=body, includes=["<cuda.h>"], return_val=f"tma_store_2d_fn(&{desc}, {smem_ptr}, {c0}, {c1})")


def codegen_tma_store_commit():
    body = """
__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tma_store_commit_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_commit)
def tma_store_arrive():
    """Compatibility name for :func:`tma_store_commit`."""
    raise RuntimeError("should not call tma_store_arrive in compilation")


# ==============================================================================
# TMA 1D Bulk Copy (used by FlashComm EP kernels)
# ==============================================================================


@builtin
class tma_copy_1d_g2s(BuiltinOp):
    """TMA 1D bulk copy from global to shared memory."""

    @staticmethod
    def return_type(gmem_ptr, mbar_ptr, smem_ptr, load_bytes):
        return void

    @staticmethod
    def codegen(gmem_ptr, mbar_ptr, smem_ptr, load_bytes):
        body = """
__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}
"""
        return Builtin(body=body, includes=[],
                       return_val=f"tma_copy_1d_g2s_fn({gmem_ptr}, {mbar_ptr}, {smem_ptr}, {load_bytes})")


@builtin
class tma_copy_1d_s2g(BuiltinOp):
    """TMA 1D bulk copy from shared to global memory."""

    @staticmethod
    def return_type(smem_ptr, gmem_ptr, store_bytes):
        return void

    @staticmethod
    def codegen(smem_ptr, gmem_ptr, store_bytes):
        body = """
__device__ __forceinline__ void tma_copy_1d_s2g_fn(void const* smem, void* gmem, int32_t bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\\n"
        :: "l"(gmem), "r"(smem_int), "r"(bytes) : "memory");
}
"""
        return Builtin(body=body, includes=[], return_val=f"tma_copy_1d_s2g_fn({smem_ptr}, {gmem_ptr}, {store_bytes})")


@builtin
class tma_store_wait_group(BuiltinOp):
    """Wait until at most ``count`` committed TMA store groups remain pending."""

    @staticmethod
    def return_type(count):
        return void

    @staticmethod
    def codegen(count):
        body = """
__device__ __forceinline__ void tma_store_wait_group_fn(int count) {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    switch (count) {
        case 0: asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory"); break;
        case 1: asm volatile("cp.async.bulk.wait_group 1;\\n" ::: "memory"); break;
        case 2: asm volatile("cp.async.bulk.wait_group 2;\\n" ::: "memory"); break;
        case 3: asm volatile("cp.async.bulk.wait_group 3;\\n" ::: "memory"); break;
        case 4: asm volatile("cp.async.bulk.wait_group 4;\\n" ::: "memory"); break;
        case 5: asm volatile("cp.async.bulk.wait_group 5;\\n" ::: "memory"); break;
        case 6: asm volatile("cp.async.bulk.wait_group 6;\\n" ::: "memory"); break;
        case 7: asm volatile("cp.async.bulk.wait_group 7;\\n" ::: "memory"); break;
        default: asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory"); break;
    }
}
"""
        return Builtin(body=body, includes=[], return_val=f"tma_store_wait_group_fn({count})")


# FlashComm compatibility; new code uses tma_store_wait_group.
tma_store_wait_n = tma_store_wait_group

# ==============================================================================
# TMA Descriptor
# ==============================================================================


@builtin
class prefetch_tma_descriptor(BuiltinOp):
    """Prefetch TMA descriptor."""

    @staticmethod
    def return_type(tensor_map):
        return void

    @staticmethod
    def codegen(tensor_map):
        body = """
__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}
"""
        return Builtin(body=body, includes=["<cuda.h>"], return_val=f"prefetch_tma_descriptor_fn(&{tensor_map})")


# ==============================================================================
# SM100 paired-SM TMA loads and multicast variants
# ==============================================================================


def _codegen_tma_load_mcast(rank, cta_group, desc, bar_ptr, smem_ptr, coords, multicast_mask, issue_pred, *,
                            peer_mask="", group_qualifier=""):
    coord_params = ", ".join(f"int32_t c{i}" for i in range(rank))
    coord_operands = ", ".join(f'"r"(c{i})' for i in range(rank))
    coord_refs = ", ".join(f"%{i + 2}" for i in range(rank))
    bar_operand = rank + 2
    mask_operand = rank + 3
    hint_operand = rank + 4
    args = ", ".join(str(coord) for coord in coords)
    body = f"""
__device__ __forceinline__ void tma_load_{rank}d_{cta_group}sm_mcast_fn(
        const CUtensorMap* d, uint64_t* bar, void* smem,
        {coord_params}, uint16_t mask, uint32_t issue_pred) {{
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = ((uint32_t)__cvta_generic_to_shared(&bar[0])){peer_mask};
    uint64_t cache_hint = 0;
    if (issue_pred != 0) {{
        asm volatile(
            "cp.async.bulk.tensor.{rank}d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes.multicast::cluster"
            ".L2::cache_hint{group_qualifier}"
            " [%0], [%1, {{{coord_refs}}}], [%{bar_operand}], %{mask_operand}, %{hint_operand};"
            :: "r"(sa), "l"((uint64_t)d), {coord_operands},
               "r"(ba), "h"(mask), "l"(cache_hint) : "memory");
    }}
}}
"""
    return Builtin(
        body=body,
        includes=["<cuda.h>"],
        return_val=(f"tma_load_{rank}d_{cta_group}sm_mcast_fn("
                    f"&{desc}, {bar_ptr}, {smem_ptr}, {args}, "
                    f"{multicast_mask}, {issue_pred})"),
    )


def codegen_tma_load_2d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask, issue_pred=1):
    return _codegen_tma_load_mcast(2, 1, desc, bar_ptr, smem_ptr, (c0, c1), multicast_mask, issue_pred)


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_2d_1sm_mcast)
def tma_load_2d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask, issue_pred=1):
    raise RuntimeError("should not call tma_load_2d_1sm_mcast in compilation")


def codegen_tma_load_3d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, multicast_mask, issue_pred=1):
    return _codegen_tma_load_mcast(3, 1, desc, bar_ptr, smem_ptr, (c0, c1, c2), multicast_mask, issue_pred)


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_3d_1sm_mcast)
def tma_load_3d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, multicast_mask, issue_pred=1):
    raise RuntimeError("should not call tma_load_3d_1sm_mcast in compilation")


def codegen_tma_load_4d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, c3, multicast_mask, issue_pred=1):
    return _codegen_tma_load_mcast(4, 1, desc, bar_ptr, smem_ptr, (c0, c1, c2, c3), multicast_mask, issue_pred)


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_4d_1sm_mcast)
def tma_load_4d_1sm_mcast(desc, bar_ptr, smem_ptr, c0, c1, c2, c3, multicast_mask, issue_pred=1):
    raise RuntimeError("should not call tma_load_4d_1sm_mcast in compilation")


def codegen_tma_store_3d(desc, smem_ptr, c0, c1, c2):
    body = """
__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(sa) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_store_3d_fn(&{desc}, {smem_ptr}, {c0}, {c1}, {c2})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_3d)
def tma_store_3d(desc, smem_ptr, c0, c1, c2):
    """TMA 3D store from shared to global memory."""
    raise RuntimeError("should not call tma_store_3d in compilation")


def codegen_tma_store_4d(desc, smem_ptr, c0, c1, c2, c3):
    body = """
__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(sa) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_store_4d_fn(&{desc}, {smem_ptr}, {c0}, {c1}, {c2}, {c3})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_4d)
def tma_store_4d(desc, smem_ptr, c0, c1, c2, c3):
    """TMA 4D store from shared to global memory."""
    raise RuntimeError("should not call tma_store_4d in compilation")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_commit)
def tma_store_commit():
    """TMA store commit group."""
    raise RuntimeError("should not call tma_store_commit in compilation")
