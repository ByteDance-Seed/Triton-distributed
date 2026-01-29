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
from triton_dist.utils import is_cuda, is_hip, is_maca
from triton.language import core
import triton.language as tl
import triton


@triton.jit
def pack_b32_v2_amd(low, high):
    return tl.cast(high, tl.uint64) << 32 | tl.cast(low, tl.uint64)


@core.extern
def pack_b32_v2_nv(low, high, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.b64 $0, {$1, $2};",
        constraints=("=l,r,r"),
        args=[low, high],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


pack_b32_v2 = pack_b32_v2_nv if is_cuda() else pack_b32_v2_amd

if is_cuda():
    from triton_dist.language.extra.cuda.language_extra import globaltimer as cuda_globaltimer, globaltimer_lo as cuda_globaltimer_lo
    from triton_dist.language.extra.cuda.language_extra import __fence as fence_nv
    from triton.language.extra.cuda import smid
elif is_hip():
    from triton_dist.language.extra.hip.language_extra import wallclock, smid
    from triton_dist.language.extra.hip.language_extra import fence as fence_amd
else:
    raise Exception("only support cuda and hip")


@core.extern
def globaltimer(_semantic=None):
    if _semantic.builder.options.backend_name == "hip":
        return wallclock(_semantic=_semantic)
    elif _semantic.builder.options.backend_name == "cuda":
        return cuda_globaltimer(_semantic=_semantic)
    else:
        tl.static_assert(False, "only cuda/hip supported", _semantic=_semantic)


@core.extern
def globaltimer_lo(_semantic=None):
    if _semantic.builder.options.backend_name == "cuda":
        return cuda_globaltimer_lo(_semantic=_semantic)
    elif _semantic.builder.options.backend_name == "hip":
        return globaltimer(_semantic=_semantic).to(tl.uint32, _semantic=_semantic)
    else:
        tl.static_assert(False, "only cuda/hip supported", _semantic=_semantic)


@core.extern
def _to_rocm_scope(scope: core.constexpr = core.constexpr("gpu"), _semantic=None):
    # convert NVIDIA scope to AMD scope
    # AMD use monotonic/acquire/release/acq_rel
    # NVIDIA use relaxed/acquire/release/acq_rel
    tl.static_assert(
        core._unwrap_if_constexpr(scope) == "cta" or core._unwrap_if_constexpr(scope) == "gpu"
        or core._unwrap_if_constexpr(scope) == "sys", "cta/gpu/sys scope is supported", _semantic=_semantic)
    if _semantic.builder.options.backend_name == "hip":
        if core._unwrap_if_constexpr(scope) == "cta":
            return "workgroup"
        elif core._unwrap_if_constexpr(scope) == "gpu":
            return "agent"
        elif core._unwrap_if_constexpr(scope) == "sys":
            return "system"
        else:
            tl.static_assert(False, scope, _semantic=_semantic)
    else:
        return scope


@core.extern
def _to_rocm_semantic(semantic: core.constexpr = core.constexpr("sc"), _semantic=None):
    # convert NVIDIA semantic to AMD semantic
    # AMD use workgroup/agent/system
    # NVIDIA use cta/gpu/sys\
    tl.static_assert(
        core._unwrap_if_constexpr(semantic) == "sc" or core._unwrap_if_constexpr(semantic) == "relaxed"
        or core._unwrap_if_constexpr(semantic) == "acquire" or core._unwrap_if_constexpr(semantic) == "release"
        or core._unwrap_if_constexpr(semantic) == "sc", _semantic=_semantic)
    if _semantic.builder.options.backend_name == "hip":
        if core._unwrap_if_constexpr(semantic) == "relaxed":
            return "monotonic"
        elif core._unwrap_if_constexpr(semantic) == "sc":
            return "seq_cst"
        else:
            return semantic
    else:
        return semantic


@core.extern
def fence(
        semantic: core.constexpr = core.constexpr("sc"), scope: core.constexpr = core.constexpr("gpu"), _semantic=None):
    # even this is a unified interface, still for NVIDIA and AMD, there are different semantic and scope
    if _semantic.builder.options.backend_name == "hip":
        return fence_amd(_to_rocm_semantic(semantic, _semantic=_semantic), _to_rocm_scope(scope, _semantic=_semantic),
                         _semantic=_semantic)
    elif _semantic.builder.options.backend_name == "cuda":
        return fence_nv(semantic, scope, _semantic=_semantic)
    else:
        tl.static_assert(False, _semantic=_semantic)


if is_cuda():
    from .nvidia.common_ops import _wait_eq_cuda as wait_eq
    from .nvidia.common_ops import _set_signal_cuda as set_signal
elif is_hip():
    from .amd.common_ops import _wait_eq_hip as wait_eq
    from .amd.common_ops import _set_signal_hip as set_signal
elif is_maca():
    from .metax.common_ops import _wait_eq_maca as wait_eq
    from .metax.common_ops import _set_signal_maca as set_signal
else:
    raise Exception("only support cuda and hip")

__all__ = ["wait_eq", "set_signal", "globaltimer", "globaltimer_lo", "pack_b32_v2", "smid"]
