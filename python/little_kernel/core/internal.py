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

__LITTLE_KERNEL_ENTRY__ = "__LITTLE_KERNEL_ENTRY__"
__INTERNAL_DYN_SHMEM__ = "__INTERNAL_DYN_SHMEM__"

# Compile-ctx key for a {var_name: (int, ...)} map recording the declared
# per-dimension extents of rank>=2 local-scope (register) arrays. Written by
# the insert_mem_alloc pass (which flattens the declaration to a 1-D C array),
# read by expression codegen to flatten multi-dimensional subscripts of those
# arrays to matching 1-D indices.
__LOCAL_ARRAY_SHAPES_CTX_KEY__ = "__LOCAL_ARRAY_SHAPES__"

# Compile-ctx key for a {var_name: {"num_elems": int, "dtype": LLType}} map
# recording typed WGMMA accumulator register fragments (semantic key
# ir.wgmma.acc_fragment). Written by the SM90 wgmma_acc_fragment declaration
# builtin's codegen (emitted before any use) and read by expression codegen
# (element subscripts lower to the per-element scalar ``<name>_<i>`` instead
# of an address-taken array slot) and by the fragment-operand WGMMA builtins
# (the accumulator operand binds the individual scalars as separate "+f" asm
# operands, mirroring hand-written CUDA, so the fragment stays in registers
# across the MMA and its elementwise epilogue).
__WGMMA_ACC_FRAGMENTS_CTX_KEY__ = "__WGMMA_ACC_FRAGMENTS__"

# Compile-ctx key for a {var_name: {"num_elems": int, "dtype": LLType}} map
# recording typed warp-level mma.sync register fragments (semantic key
# ir.mma.m16n8k16.bf16.per_thread_fragment). Written by the NV-common
# mma_acc_fragment_f32 / mma_operand_fragment_u32 declaration builtins
# (emitted before any use) and read by expression codegen (element subscripts
# lower to the per-element scalar ``<name>_<i>`` at compile-time-unrolled
# indices, never an address-taken array slot) and by the fragment-aware
# mma.sync builtins (each operand binds its individual scalars as separate
# asm operands, so the fragments stay in registers across the MMA and any
# elementwise prologue/epilogue). Distinct from the WGMMA key: warp-level
# mma.sync fragments cover accumulator (f32, 4/thread on m16n8k16) AND the
# packed 16-bit A/B operand registers (u32), while WGMMA accumulators are
# f32-only and sized for m64nNk16.
__MMA_SYNC_FRAGMENTS_CTX_KEY__ = "__MMA_SYNC_FRAGMENTS__"
