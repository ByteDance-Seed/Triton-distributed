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
import triton_dist.language.extra.cuda.language_extra as cuda_language_extra
import triton_dist.language.extra.hip.language_extra as hip_language_extra
from triton_dist.utils import is_cuda, is_hip
from triton.language import core
from .utils import ModuleProxy

_extra_module = ModuleProxy([
    (is_cuda, cuda_language_extra),
    (is_hip, hip_language_extra),
])


@_extra_module.dispatch
def tid(axis):
    ...


@_extra_module.dispatch
def atomic_cas(ptr, cmp_val, target_val, scope, semantic):
    ...


@_extra_module.dispatch
def atomic_add(ptr, val, scope, semantic):
    ...


@_extra_module.dispatch
def __syncthreads():
    ...


@_extra_module.dispatch
def ld(ptr, scope, semantic):
    ...


@_extra_module.dispatch
def st(ptr, val, scope, semantic):
    ...


@core.builtin
def threads_per_warp(_semantic=None):
    if is_cuda():
        return core.constexpr(32)
    elif is_hip():
        return core.constexpr(64)
    else:
        raise NotImplementedError("unsupported")


@core.builtin
def num_threads(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps * threads_per_warp(_semantic=_semantic))


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)


__all__ = [
    "atomic_cas",
    "atomic_add",
    "__syncthreads",
    "tid",
    "ld",
    "st",
    "num_threads",
    "num_warps",
    "threads_per_warp",
]
