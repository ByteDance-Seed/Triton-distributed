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

import functools
from typing import Any, Callable, Hashable, Optional, Tuple

import torch

from flash_comm.utils import KernelHandle, KernelSpec, get_global_cutedsl_kernel_cache

# Kernel geometry constants for the M-contig GEMM family. These mirror
# the hard-coded ``mma_tiler`` / ``cluster_shape`` inside the kernels
# under ``flash_comm.ep_overlap.kernels.*`` and are NOT user-tunable.
# ``GEMM_CLUSTER_TILE_M`` is the per-expert M-tile callers must pad to.
_GEMM_MMA_TILER_M: int = 256
_GEMM_CLUSTER_M: int = 2
_GEMM_CTA_GROUP: int = 2  # use_2cta_instrs=True throughout
GEMM_CLUSTER_TILE_M: int = (_GEMM_MMA_TILER_M // _GEMM_CTA_GROUP) * _GEMM_CLUSTER_M


def mark_dynamic(tensor: torch.Tensor, *, assumed_align: Optional[int] = None, enable_tvm_ffi: bool = False):
    """Wrap a tensor and mark its layout dynamic for CuTeDSL compile."""
    from cutlass.cute.runtime import from_dlpack

    kwargs = {}
    if assumed_align is not None:
        kwargs["assumed_align"] = assumed_align
    if enable_tvm_ffi:
        kwargs["enable_tvm_ffi"] = True
    return from_dlpack(tensor, **kwargs).mark_layout_dynamic()


def make_moe_jit_dummies(*, num_experts: int, n: int, hidden_in: int, ab_dtype: torch.dtype,
                         c_dtype: torch.dtype = None):
    """Minimal-allocation placeholders for the M-contig MoE group-GEMM JIT.

    The kernels treat the (M, K) / (N, K, L) / (M, N) tensors as
    layout-dynamic; the JIT only inspects rank / dtype / which mode has
    stride==1. Actual element values and the dynamic-axis sizes are never
    read. We therefore allocate the minimum that still pins the kernel's
    leading-stride-1 invariant:

      A   : (1, K)    row-major,           ~K  elements
      B   : (1, K, L) K-leading via permute (L, 1, K) -> (1, K, L);
                       contiguous storage is L*K elements
      C   : (1, N)    row-major,           ~N  elements
      psm : (L,)      Int32 CUDA tensor matching the runtime
                       ``recv_expert_counts`` / ``problem_sizes_m`` ABI.

    Production ``setup_moe_problem`` allocates ``M_padded * (K + N)`` for
    A+C and ``L * N * K`` for B; with K=8192 / N=5120 / L=160 that is
    >13 GB of BF16 just to feed ``cute.compile``. This helper drops the
    total to a few MB and runs in microseconds.
    """
    if c_dtype is None:
        c_dtype = ab_dtype
    a = torch.empty((1, hidden_in), dtype=ab_dtype, device="cuda")
    # (L, 1, K).permute(1, 2, 0) -> (1, K, L) with strides (K, 1, K); the
    # stride==1 on mode 1 selects K as the leading dim, matching the
    # production view from ``B_ref.permute(1, 2, 0)``.
    b = (torch.empty((num_experts, 1, hidden_in), dtype=ab_dtype, device="cuda").permute(1, 2, 0))
    c = torch.empty((1, n), dtype=c_dtype, device="cuda")
    psm = torch.empty(num_experts, dtype=torch.int32, device="cuda")
    return a, b, c, psm


def mark_compact_dynamic(tensor: torch.Tensor, *, assumed_align: int = 16, mode: int = 0, divisibility: int = 1,
                         enable_tvm_ffi: bool = False):
    """Compact-shape-dynamic wrap for vectorised 128b SIMT atoms.

    Preserves the leading-stride-1 + 16 B row alignment invariants the
    reduce kernels need while letting the outer (token-axis) shape vary.
    """
    from cutlass.cute.runtime import from_dlpack

    kwargs = {"assumed_align": assumed_align}
    if enable_tvm_ffi:
        kwargs["enable_tvm_ffi"] = True
    return from_dlpack(tensor, **kwargs).mark_compact_shape_dynamic(
        mode=mode,
        divisibility=divisibility,
    )


def cute_compile_options(*extra_options: str, enable_tvm_ffi: bool = False, opt_level: Optional[int] = None) -> str:
    """Build a CuTe compile option string."""
    options = []
    if enable_tvm_ffi:
        options.append("--enable-tvm-ffi")
    if opt_level is not None:
        options.extend(["--opt-level", str(int(opt_level))])
    for option in extra_options:
        if option:
            options.extend(str(option).split())
    return " ".join(options)


class CuTeDSLEPOverlapOpBase:
    """Common base for every CuTeDSL EP-overlap op."""

    def __init__(self, *, rank: int, world_size: int):
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._cache = get_global_cutedsl_kernel_cache()
        cls = type(self)
        self._op_name = f"{cls.__module__}.{cls.__qualname__}"

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @staticmethod
    def _assert_supported_dtype(dtype: torch.dtype, op_name: str) -> None:
        if dtype not in {torch.bfloat16, torch.float16}:
            raise NotImplementedError(f"{op_name} does not support dtype {dtype}")

    def _get_cached_kernel(self, *, variant_args: Hashable, builder: Callable[[], Any],
                           compile_options: Optional[Tuple[str, ...]] = None) -> KernelHandle:
        if compile_options is None:
            compile_options = ("--enable-tvm-ffi", )
        spec = KernelSpec(
            op_name=self.op_name,
            variant_args=variant_args,
            builder=builder,
            compile_options=compile_options,
        )
        return self._cache.get_or_compile(spec)


@functools.lru_cache(maxsize=1)
def resolve_nvcc_opt_level() -> int:
    """CUDA-version aware ``--opt-level`` for the M-contig GEMM kernels.

    CUDA 13.1+ regresses at level 3 on these kernels; older drivers
    prefer level 3. Falls back to 3 when ``CUDA_VERSION`` is unavailable.
    """
    try:
        from cutlass import CUDA_VERSION
        if CUDA_VERSION.major < 13:
            return 3
        if CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1:
            return 3
        return 2
    except ImportError:
        return 3
