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
"""Compatibility facade for the vibe-cuda-eval exporter.

RF09 moved schema ownership, shape expressions, launch/alias contracts, CUDA
emission and host bindings under the generic :mod:`little_kernel.backends.nv.common.export` API.
This integration module is intentionally thin: it re-exports the canonical
symbols existing tests and the evaluator corpus import privately. New code
must depend on ``little_kernel.backends.nv.common.export`` (load_spec/validate/export), not on
these underscore names.
"""

from __future__ import annotations

from little_kernel.backends.nv.common.export._bindings import _render_binding, _render_model
from little_kernel.backends.nv.common.export._cuda import _render_cuda
from little_kernel.backends.nv.common.export._foreach import _FOREACH_FUSED_MAX, _render_foreach_fused_cuda, _render_foreach_single_binding, _render_foreach_single_cuda, _validate_foreach_single
from little_kernel.backends.nv.common.export._frontend import _cuda_kernel_arg, _declared_tensor_dtypes, _descriptor_param_kind, _kernel_param_names, _launcher_arg_decl, _load_kernel, _scalar_splits
from little_kernel.backends.nv.common.export._launch import _validate_launch
from little_kernel.backends.nv.common.export._models import validate_spec as _validate_spec
from little_kernel.backends.nv.common.export._runner import check, export, main
from little_kernel.backends.nv.common.export._schema import _CONSTRUCTOR_CPP, _DTYPES, _IDENTIFIER, _SCALAR_COMPONENT, _SCALAR_DTYPES, _SCALAR_RUNTIME_CPP, _TMA_DESCRIPTOR_FIELDS, _TMA_DTYPES, _TMA_SWIZZLES, _identifier, _positive_int
from little_kernel.backends.nv.common.export._shapes import _derived_constexpr_expr, _flattened_leading_tma_sources, _shape_expr, _split_cuda_prelude
from little_kernel.backends.nv.common.export._spec import _load_spec

__all__ = [
    "_DTYPES",
    "_TMA_DTYPES",
    "_TMA_SWIZZLES",
    "_TMA_DESCRIPTOR_FIELDS",
    "_CONSTRUCTOR_CPP",
    "_SCALAR_DTYPES",
    "_SCALAR_COMPONENT",
    "_SCALAR_RUNTIME_CPP",
    "_IDENTIFIER",
    "_FOREACH_FUSED_MAX",
    "_identifier",
    "_positive_int",
    "_flattened_leading_tma_sources",
    "_shape_expr",
    "_derived_constexpr_expr",
    "_split_cuda_prelude",
    "_validate_launch",
    "_load_spec",
    "_load_kernel",
    "_declared_tensor_dtypes",
    "_scalar_splits",
    "_launcher_arg_decl",
    "_descriptor_param_kind",
    "_cuda_kernel_arg",
    "_kernel_param_names",
    "_validate_foreach_single",
    "_render_foreach_fused_cuda",
    "_render_foreach_single_cuda",
    "_render_foreach_single_binding",
    "_render_cuda",
    "_render_binding",
    "_render_model",
    "export",
    "check",
    "main",
    "_validate_spec",
]

if __name__ == "__main__":
    raise SystemExit(main())
