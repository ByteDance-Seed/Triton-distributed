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
"""Per-thread FP32 dot-accumulate of a strided GMEM column against a
shared-memory row (``ir.warp_smem_block_dot.accumulate_f32``).

Recurrent/decode state-update kernels (chunked linear attention, gated
delta rules, LRU/SSM scans) share a recurring inner shape: a block stages a
[*, K] fp32 operand row (normalized keys, queries, gates) in shared memory,
and EVERY thread owns a private accumulator plus one strided column of a
[K, V]-shaped DRAM state tensor.  The per-thread epilogue is then

    acc += sum_{i < K} col[base + i * stride] * s_row[row_off + i]

expressed hand-rolled today as a runtime loop over ``gmem_ld_scalar`` and
``fma``.  That hand-rolled form has two structural weaknesses the IR cannot
repair locally:

* The loop's gmem access pattern (unit-column, ``stride``-strided, one
  element per k-step) is invisible to any analysis that wants to reason
  about DRAM traffic; every iteration looks like an independent scalar
  load, so passes cannot prove the column is read exactly once.
* When the per-k-step value must ALSO update a private state row (the
  teacher decomposition in these kernels), frontends reach for a K-length
  per-thread array with runtime row indexing.  ``alloc_local_memory``
  provides exactly that signature, but a dynamically-indexed local array
  is placed in LOCAL MEMORY (spilled), not registers: the index is not
  statically resolvable, so no allocation pass may promote it.  The
  recurrence then pays a local-memory round-trip per element per step.

``smem_dot_accumulate_f32`` removes the need for that register array by
giving the IR a first-class typed primitive for the dot itself: the state
column streams from DRAM exactly once through the FMA chain, so no
K-element dynamically-indexed array ever needs to exist.  The intrinsic is
deliberately generic — it knows nothing about decays, gates, or any
particular update rule; callers fold those scalars into ``acc`` or the
staged row with ordinary arithmetic before/after the call.

Typing and legality contract:

* ``acc`` must be fp32 (the accumulation dtype; the semantic key for this
  family is fp32-only, matching the fp32 state rows these kernels keep in
  registers even when outer activations are lower precision).
* ``gmem`` must be a Tensor whose element type is fp32; ``smem`` likewise.
  Mixed-precision dots are intentionally NOT folded into this op: an
  fp16/bf16 variant with fp32 accumulate is a separate typed intrinsic
  (different FMA contraction rules), not a silent conversion here.
* ``gmem_base``, ``gmem_stride``, ``smem_base`` are element offsets
  (int32/int64 expressions); ``count`` is the dot length.  All may be
  runtime values; when ``count`` is a compile-time constant the emitted
  helper is fully unrollable (``#pragma unroll`` is emitted on the loop,
  and nvcc unrolls constant-trip loops inside ``__forceinline__``
  helpers).
* Cache policy: same vocabulary as ``gmem_ld_scalar`` — "wb" (default),
  "ca", "cg", "cs", "lu", "nc", "cv".  This is a pure cache-allocation
  hint on the DRAM operand; bit-exact regardless of choice.  "nc"
  (__ldg) carries its usual read-only-for-the-launch contract, asserted
  by the frontend.  Each policy lowers to its OWN distinct helper
  (``lk_smem_dot_accumulate_f32[_<policy>]_fn``), so one kernel may use
  several policies without any call site silently inheriting another
  site's load spelling.

Declared memory effects:

* ``Effect(r(GMEM), "gmem")``: the op reads ``count`` strided elements
  from the DRAM tensor and must not be treated as side-effect free or
  moved across a conflicting write to the same allocation.
* ``Effect(r(SMEM), "smem")``: the op reads ``count`` consecutive staged
  elements; effect analyses (dataflow mem-event collection, race checks)
  must see the shared row read so a ``__syncthreads`` boundary staged
  before the op is never reordered after it.

The smem read is performed by each participating thread independently
(broadcast within a warp when lanes share the row); there is no cross-lane
data movement, no barrier, and no convergence requirement — this is a
pure per-thread compute op over two memory operands, composable inside
any control flow.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import float32, TensorType
from little_kernel.core.memory_effects import MS, r
import little_kernel.backends.nv.common.language.intrin.contracts as contracts
from little_kernel.backends.nv.common.language.intrin.memory_ops.scalar_global import (
    _GMEM_LD_SCALAR_INTRIN,
    _normalize_cache_policy,
)

_SMEM_DOT_EFFECTS = (
    # DRAM state column: ``count`` strided fp32 elements, read once.
    r(MS.GMEM, "gmem"),
    # Staged shared row: ``count`` consecutive fp32 elements, read by
    # every participating thread (warp-broadcast when lanes share rows).
    r(MS.SMEM, "smem"),
)


def _check_f32_tensor(dtype, name):
    if dtype is None:
        raise TypeError(f"smem_dot_accumulate_f32: cannot infer type of {name}")
    if not isinstance(dtype, TensorType):
        raise TypeError(f"smem_dot_accumulate_f32 expects {name} to be a Tensor, got "
                        f"{type(dtype).__name__}: {dtype}")
    if not dtype.element_type == float32:
        raise TypeError(f"smem_dot_accumulate_f32 expects fp32 elements for {name}, "
                        f"got {dtype.element_type!r}.  Cast to fp32 explicitly; "
                        "mixed-precision dots are a separate typed intrinsic.")


def _eval_return_type_smem_dot(acc_dtype, gmem_dtype, gmem_base_dtype, gmem_stride_dtype, smem_dtype, smem_base_dtype,
                               count_dtype, policy=None):
    """Return fp32; validate that acc/gmem/smem are fp32-typed."""
    if acc_dtype is not None and not acc_dtype == float32:
        raise TypeError("smem_dot_accumulate_f32 expects an fp32 accumulator, got "
                        f"{acc_dtype!r}.  Cast to fp32 explicitly; the fp32 accumulate "
                        "contract is deliberate (see module docstring).")
    _check_f32_tensor(gmem_dtype, "gmem")
    _check_f32_tensor(smem_dtype, "smem")
    return float32


_SMEM_DOT_BODY = """
#ifndef {guard}
#define {guard}
__device__ __forceinline__ float {fn_name}(
        float acc,
        const float* lk_gmem,
        long long lk_gmem_base,
        long long lk_gmem_stride,
        const float* lk_smem,
        long long lk_smem_base,
        int lk_count) {{
    // Per-thread dot of one strided DRAM column against a staged shared
    // row.  The gmem operand streams exactly once: each iteration issues
    // one policy-selected scalar load and one FMA into the register
    // accumulator, so no K-length dynamically-indexed private array ever
    // needs to exist (such an array would spill to local memory).
    //
    // The load expression at the gmem site is substituted per
    // cache-policy at codegen time; the smem side is a plain subscript
    // (shared memory has no cache-policy vocabulary).  #pragma unroll
    // lets nvcc fully unroll constant-trip uses inside this
    // __forceinline__ helper.
#pragma unroll
    for (int lk_i = 0; lk_i < lk_count; ++lk_i) {{
        const float lk_col = {ld_expr};
        acc = fmaf(lk_col, lk_smem[lk_smem_base + lk_i], acc);
    }}
    return acc;
}}
#endif
"""


def codegen_smem_dot_accumulate_f32(acc, gmem, gmem_base, gmem_stride, smem, smem_base, count, policy=None,
                                    _arg_dtypes=None):
    policy = _normalize_cache_policy(policy, _GMEM_LD_SCALAR_INTRIN.keys(), "wb", "smem_dot_accumulate_f32")
    intrin = _GMEM_LD_SCALAR_INTRIN[policy]
    # "wb" (default) has no __ld* spelling; a plain dereference is the
    # bit-exact write-back load.  Other policies reuse the same CUDA
    # cache-operator intrinsics as gmem_ld_scalar.
    if intrin is None:
        # Parenthesize the address expression: a bare `*ptr + off` would
        # bind the dereference to the pointer alone.
        ld_expr = "(*(lk_gmem + (lk_gmem_base + (long long)lk_i * lk_gmem_stride)))"
    else:
        ld_expr = (f"{intrin}(lk_gmem + (lk_gmem_base + (long long)lk_i * "
                   "lk_gmem_stride))")
    # The helper body is policy-specialized, so the helper NAME must be
    # policy-distinct: writeln_builtin deduplicates builtin bodies by the
    # extracted C++ function name, and a policy-invariant name would let
    # the first-emitted policy's body absorb later call sites that asked
    # for a different policy (silently dropping the hint, or worse,
    # silently inheriting an __ldg "nc" body at a read/write site).
    # Same per-policy suffix + #ifndef-guard pattern as codegen_gmem_ld_v4.
    suffix = "" if policy == "wb" else f"_{policy}"
    fn_name = f"lk_smem_dot_accumulate_f32{suffix}_fn"
    guard = f"LK_SMEM_DOT_ACCUMULATE_F32{suffix.upper()}_FN_DEFINED"
    body = _SMEM_DOT_BODY.format(guard=guard, fn_name=fn_name, ld_expr=ld_expr)
    return Builtin(
        body=body,
        includes=["<device_functions.h>"],
        return_val=(f"{fn_name}({acc}, {gmem}, {gmem_base}, "
                    f"{gmem_stride}, {smem}, {smem_base}, {count})"),
    )


@builtin(eval_return_type=_eval_return_type_smem_dot, codegen_func=codegen_smem_dot_accumulate_f32,
         effects=_SMEM_DOT_EFFECTS)
def smem_dot_accumulate_f32(acc, gmem, gmem_base, gmem_stride, smem, smem_base, count, policy=None):
    """Per-thread fp32 dot-accumulate: strided GMEM column x shared row.

    Returns

        acc + sum_{i < count} gmem[gmem_base + i * gmem_stride]
                                * smem[smem_base + i]

    computed as a serial FMA chain in the thread's registers.  Both memory
    operands are fp32 tensors; ``acc`` is fp32.  Offsets are element
    offsets and may be runtime values; ``count`` may be runtime or
    compile-time constant (constant-trip uses fully unroll).

    Use this whenever a block stages an fp32 operand row in shared memory
    and each thread must contract it against a strided column of a DRAM
    tensor (recurrent state columns, decode value columns, per-output
    epilogue dots) -- the pattern that otherwise forces either a
    dynamically-indexed local array (which spills) or an opaque scalar
    load loop (which hides the single-pass DRAM traffic shape from
    analysis).

    ``policy`` is the same compile-time cache-policy string as
    ``gmem_ld_scalar`` ("wb" default, "ca", "cg", "cs", "lu", "nc", "cv")
    and applies to the DRAM operand only; it is bit-exact.  "nc" carries
    the __ldg read-only-for-the-launch contract, asserted by the frontend.
    """
    raise RuntimeError("smem_dot_accumulate_f32 should never be called in compilation")


setattr(smem_dot_accumulate_f32, CODEGEN_NEEDS_TYPES_ATTR, True)
