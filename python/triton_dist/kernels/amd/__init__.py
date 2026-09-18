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
import importlib

from .ep_a2a_intra_node import (
    kernel_dispatch_token_intra_node,
    kernel_skipped_token_local_dispatch_intra_node,
    kernel_skipped_token_inplace_local_combine_intra_node,
    kernel_combine_token_intra_node,
    get_ag_splits_and_recv_offset_for_dispatch_intra_node,
)
from .low_latency_all_to_all import create_all_to_all_context, fast_all_to_all, all_to_all_post_process

# The AG-GEMM / GEMM-RS kernels are rocshmem-specific (they import ``pyrocshmem``
# at module load). Import them lazily (PEP 562) so that merely importing this
# package -- e.g. for a backend-agnostic kernel like ``common_ops`` (grid
# barriers) -- does NOT drag in a hard ``pyrocshmem`` dependency. Accessing an
# AG-GEMM/GEMM-RS symbol still loads its module (and requires rocshmem), which is
# the correct behaviour: those kernels are unusable without the rocshmem backend.
_LAZY_SUBMODULES = {
    "ag_gemm_intra_node": "allgather_gemm",
    "create_ag_gemm_intra_node_context": "allgather_gemm",
    "gemm_rs_intra_node": "gemm_reduce_scatter",
    "create_gemm_rs_intra_node_context": "gemm_reduce_scatter",
}

__all__ = list(_LAZY_SUBMODULES) + [
    "kernel_dispatch_token_intra_node",
    "kernel_skipped_token_local_dispatch_intra_node",
    "kernel_skipped_token_inplace_local_combine_intra_node",
    "kernel_combine_token_intra_node",
    "get_ag_splits_and_recv_offset_for_dispatch_intra_node",
    "create_all_to_all_context",
    "fast_all_to_all",
    "all_to_all_post_process",
]

# The fused intra-node EP dispatch/combine kernels are mori_shmem-based; import
# them eagerly but tolerate a missing mori_shmem backend (warn, don't fail) so
# the package stays importable on non-AMD hosts.
try:
    from .ep_all2all_fused import (
        create_ep_a2a_fused_context,
        fused_dispatch_token_moe_grouped_gemm,
        fused_group_gemm_combine_token,
        mega_kernel_dispatch_token_moe_grouped_gemm,
        mega_kernel_moe_grouped_gemm_combine_token,
        kernel_build_fused_dispatch_metadata,
        kernel_build_gemm_tiling,
    )
    __all__ += [
        "create_ep_a2a_fused_context",
        "fused_dispatch_token_moe_grouped_gemm",
        "fused_group_gemm_combine_token",
        "mega_kernel_dispatch_token_moe_grouped_gemm",
        "mega_kernel_moe_grouped_gemm_combine_token",
        "kernel_build_fused_dispatch_metadata",
        "kernel_build_gemm_tiling",
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"ep_all2all_fused unavailable (mori_shmem not installed): {e}")


def __getattr__(name):
    submod = _LAZY_SUBMODULES.get(name)
    if submod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submod}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
