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
"""Queryable tensor-core tile descriptions (coverage.mma_pipeline).

The WGMMA/UMMA intrinsics in ``little_kernel.language.intrin`` know how to
*emit* PTX for a specific legal tensor-core configuration; this module
describes the *shape of* a configuration so audit/pipeline code can derive
quantities (accumulator registers per thread, fragment element counts,
shared-memory bytes per K-step) without hardcoding PTX facts at every use
site. Every derived number is tied to the PTX contract the emitters already
enforce, so a coverage audit can check them against codegen in one place.

Legality model (cw.codegen.tilelang-ptx-inline-assembly): shape/dtype
arguments are validated against an explicit whitelist BEFORE any derived
quantity is computed; illegal configurations raise ValueError, matching the
emitter-side behaviour ("illegal config -> ValueError, never a malformed
template").
"""

from dataclasses import dataclass

import little_kernel.core.type_system as ll_type

# PTX defines `wgmma.mma_async.sync.aligned.m64nNk16.f32.{f16,bf16}.{f16,bf16}`
# for every N in {8, 16, ..., 256} (multiples of 8); the emitter-side
# whitelist lives in language.intrin.wgmma._WGMMA_LEGAL_NCOLS and must stay in
# lockstep with this one. Kept as a module-level constant so audit code can
# iterate the full legal-N range without importing the emitter internals.
WGMMA_F16_LEGAL_NCOLS = tuple(range(8, 257, 8))

# Operand dtypes the PTX m64nNk16 f32-accumulator WGMMA accepts.
WGMMA_F16_LEGAL_OPERAND_DTYPES = (ll_type.float16, ll_type.bfloat16)

# A WGMMA instruction is warpgroup-wide: 128 threads = 4 warps.
WGMMA_THREADS_PER_WARPGROUP = 128


def _check_operand_dtype(dtype, name):
    if dtype not in WGMMA_F16_LEGAL_OPERAND_DTYPES:
        raise ValueError(f"{name}: operand dtype must be one of "
                         f"{[str(d) for d in WGMMA_F16_LEGAL_OPERAND_DTYPES]} for "
                         f"m64nNk16 f32-accumulator WGMMA (got {dtype!r})")
    return dtype


def _check_mma_dims(M, K, N_legal, name):
    """Validate raw M/N/K ints against one PTX shape's contract."""
    for label, v in (("M", M), ("K", K)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name}: {label} must be a positive int (got {v!r})")
    if not isinstance(N_legal, tuple) or not N_legal:
        raise ValueError(f"{name}: N_legal must be a non-empty tuple")
    return True


@dataclass(frozen=True)
class MMA:
    """A single warpgroup-level MMA instruction shape (M x N x K)."""
    M: int
    N: int
    K: int
    A_dtype: ll_type.LLType
    B_dtype: ll_type.LLType
    ACC_dtype: ll_type.LLType

    @property
    def acc_elems_total(self) -> int:
        """Total accumulator elements produced by one MMA (M*N)."""
        return self.M * self.N

    @property
    def a_elems_total(self) -> int:
        """A-operand elements consumed per K-step (M*K)."""
        return self.M * self.K

    @property
    def b_elems_total(self) -> int:
        """B-operand elements consumed per K-step (K*N)."""
        return self.K * self.N


@dataclass(frozen=True)
class WgMMA(MMA):
    """A Hopper WGMMA shape with shared-memory operand residence flags.

    ``A_in_SMEM``/``B_in_SMEM`` record whether the operand is read from
    shared memory (SS path) or from registers (RS path) — this is the
    shared-memory pipeline axis the audit surface cares about.
    """
    A_in_SMEM: bool
    B_in_SMEM: bool

    @property
    def acc_regs_per_thread(self) -> int:
        """Accumulator registers each of the 128 warpgroup threads holds.

        PTX contract for m64nNk16 f32-accumulator WGMMA: the M*N f32
        accumulator is spread across the 128 warpgroup threads, matching the
        ``n // 2`` register count the wgmma emitter tests assert for every
        legal N (M=64 -> M*N/128 = N/2).
        """
        if self.ACC_dtype != ll_type.float32:
            raise ValueError(f"acc_regs_per_thread: only the f32-accumulator PTX WGMMA "
                             f"contract is modelled (got ACC_dtype={self.ACC_dtype!r})")
        return self.acc_elems_total // WGMMA_THREADS_PER_WARPGROUP

    @property
    def smem_bytes_per_k_step(self) -> int:
        """SMEM bytes one K-step consumes for operands resident in SMEM.

        Includes only the operands flagged as SMEM-resident (SS vs RS).
        Element widths come from the dtype itself so widening to fp8 operands
        stays a local change.
        """
        total = 0
        if self.A_in_SMEM:
            total += self.a_elems_total * (self.A_dtype.bits // 8)
        if self.B_in_SMEM:
            total += self.b_elems_total * (self.B_dtype.bits // 8)
        return total


def wgmma_m64nNk16_f32(N, A_dtype=ll_type.bfloat16, B_dtype=ll_type.bfloat16, A_in_SMEM=True, B_in_SMEM=True):
    """Build a WgMMA tile for PTX m64nNk16 f32-accumulator WGMMA.

    N is validated against the full PTX whitelist {8, 16, ..., 256} — the
    same set the wgmma_mma_async_ss/rs dispatchers accept — so audit code
    sees exactly the legal-tiling surface the emitters expose. Operand dtypes
    are restricted to f16/bf16 per the PTX contract for this accumulator.
    """
    # N is a PTX shape immediate: require a true integer type up front (no
    # strings, bools, or non-integral floats), then whitelist-check the value.
    if isinstance(N, bool) or isinstance(N, str) or not isinstance(N, (int, float)):
        raise ValueError(f"wgmma_m64nNk16_f32: N is a PTX shape immediate and must be a "
                         f"compile-time integer in [8, 256] step 8 (got {N!r})")
    try:
        parsed_n = int(N)
    except (ValueError, TypeError):
        raise ValueError(f"wgmma_m64nNk16_f32: N is a PTX shape immediate and must be a "
                         f"compile-time integer in [8, 256] step 8 (got {N!r})")
    if isinstance(N, float) and N != parsed_n:
        raise ValueError(f"wgmma_m64nNk16_f32: N must be an integer, not {N!r}")
    if parsed_n not in WGMMA_F16_LEGAL_NCOLS:
        raise ValueError(f"wgmma_m64nNk16_f32: illegal WGMMA tile N {parsed_n}; PTX "
                         "defines m64nNk16 for N a multiple of 8 in [8, 256] (f16/bf16 "
                         "operands, f32 accumulator)")
    _check_operand_dtype(A_dtype, "wgmma_m64nNk16_f32(A_dtype)")
    _check_operand_dtype(B_dtype, "wgmma_m64nNk16_f32(B_dtype)")
    for label, flag in (("A_in_SMEM", A_in_SMEM), ("B_in_SMEM", B_in_SMEM)):
        if not isinstance(flag, bool):
            raise ValueError(f"wgmma_m64nNk16_f32: {label} must be a bool (got {flag!r})")
    return WgMMA(M=64, N=parsed_n, K=16, A_dtype=A_dtype, B_dtype=B_dtype, ACC_dtype=ll_type.float32,
                 A_in_SMEM=A_in_SMEM, B_in_SMEM=B_in_SMEM)


def WgMMA_64_X_16_F32BF16BF16_SS(N: int):
    """Legacy name kept for existing callers; see wgmma_m64nNk16_f32."""
    assert N in [8 * i for i in range(1, 33)]
    return wgmma_m64nNk16_f32(N, A_dtype=ll_type.bfloat16, B_dtype=ll_type.bfloat16, A_in_SMEM=True, B_in_SMEM=True)
