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
"""Compatibility re-export of the canonical memory intrinsics (RF10).

This module used to be a divergent partial copy of
``language/intrin/memory.py`` (50 shared top-level definitions, of which 13 had
drifted: older allocation codegens without identifier quote-stripping, shorter
allocation signatures and an older 128-bit load/store spelling).  A static
audit found no in-tree importer of ``little_kernel.backends.nv.common.codegen.memory_compat``.

Per the RF10 contract there must be one maintained implementation per
operation.  This module is now a thin, documented compatibility facade that
re-exports the canonical definitions from
:mod:`little_kernel.backends.nv.common.language.intrin.memory`.  It deliberately does NOT keep
the stale divergent copies: the canonical versions are the supersets
(e.g. they strip string-literal quotes from generated identifiers and accept
the extended allocation keyword), so re-exporting them preserves the public
names while removing the second, drifting implementation.

Import policy: new code should import from
``little_kernel.backends.nv.common.language.intrin.memory`` (or the owner modules in
``little_kernel.backends.nv.common.language.intrin.memory_ops``).  This path is retained only for
out-of-tree callers that used to depend on the old codegen copy.
"""
import little_kernel.backends.nv.common.language.intrin.memory as _canonical_memory
from little_kernel.backends.nv.common.language.intrin.memory import *

# Explicit re-export of the names the historical codegen.memory module defined,
# plus the private helper spellings some callers used.
from little_kernel.backends.nv.common.language.intrin.memory import codegen_alloc_dynamic_shared_memory, alloc_dynamic_shared_memory_eval_arg_type, alloc_dynamic_shared_memory, codegen_alloc_local_memory, alloc_local_memory_eval_arg_type, alloc_local_memory, codegen_alloc_shared_memory, alloc_shared_memory_eval_arg_type, alloc_shared_memory, codegen_slice_dynamic_shared_memory, slice_dynamic_shared_memory_eval_arg_type, slice_dynamic_shared_memory, codegen_align_memory, align_memory, codegen_empty, empty, _ZERO_LITERAL, _INIT_LIST_THRESHOLD, codegen_alloc_local_zeros, alloc_local_zeros_eval_arg_type, alloc_local_zeros, codegen_zeros, zeros, codegen_alloc_local_full, alloc_local_full_eval_arg_type, alloc_local_full, codegen_full, full, codegen_ldcg, eval_return_type_ldcg, __ldcg, codegen_ldca, eval_return_type_ldca, __ldca, codegen_smem_bf16_ptr, smem_bf16_ptr, codegen_smem_u64_ptr, smem_u64_ptr, codegen_ptr_byte_offset, ptr_byte_offset, codegen_cvta_generic_to_shared, cvta_generic_to_shared, codegen_stsm_x2_from_floats, stsm_x2_from_floats, codegen_stmatrix_x4, stmatrix_x4, codegen_load_128bit, load_128bit, codegen_store_128bit, store_128bit

# The canonical module object is exposed for code that did
# ``import little_kernel.backends.nv.common.codegen.memory_compat as m`` and reached for arbitrary
# attributes (every memory intrinsic is available on it).
canonical_memory = _canonical_memory

__all__ = [name for name in dir(_canonical_memory) if not name.startswith("__")]
