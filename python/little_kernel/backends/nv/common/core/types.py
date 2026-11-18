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

from __future__ import annotations
from typing import Optional
from little_kernel.core.type_system import LLType


class TmaDescriptorType(LLType):
    """TMA (Tensor Memory Accelerator) descriptor type (singleton).

    A bare ``TmaDescriptor`` entry-parameter annotation is the documented
    shorthand for the by-value grid-constant singleton transport: CUDA
    requires a ``__global__`` CUtensorMap parameter to be declared
    ``const __grid_constant__``, so the frontend resolves the bare form to
    ``grid_constant[TmaDescriptor]`` via
    :func:`normalize_entry_grid_constant_param` (matching what the runtime
    ABI already assumes for this type). Device-array bundles keep the
    explicit ``ptr[TmaDescriptor]`` spelling; non-entry (device) functions
    keep the annotation as written.
    """
    __slots__ = ()
    _instance: Optional["TmaDescriptorType"] = None

    def __new__(cls) -> "TmaDescriptorType":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._freeze(())
            cls._instance = obj
        return cls._instance

    def __str__(self) -> str:
        return "TmaDescriptor"

    def __repr__(self) -> str:
        return "TmaDescriptor"


TmaDescriptor = TmaDescriptorType()


def _is_grid_constant_shortcut(lltype) -> bool:
    """True when a bare annotation demands the grid-constant entry transport.

    Predicate on the type value, not its name: future by-value opaque
    grid-constant structs opt in by registering here.
    """
    return isinstance(lltype, TmaDescriptorType)


def normalize_entry_grid_constant_param(lltype):
    """Resolve a bare shorthand entry-parameter type to its grid-constant form.

    ``TmaDescriptor`` -> ``const[grid_constant[TmaDescriptor]]`` (and a
    ``const[TmaDescriptor]`` wrapping gains the grid-constant marker inside,
    never a second const); every other type is returned unchanged. The
    wrapped form keeps the bare shorthand and the explicit
    ``const[grid_constant[...]]`` spelling byte-identical in the emitted
    CUDA. Shared by the CUDA signature emitter and the export manifest
    validator so both agree on the ONE accepted spelling.
    """
    from little_kernel.core.type_system import Const, GridConstant
    if _is_grid_constant_shortcut(lltype):
        return Const(GridConstant(lltype))
    if (isinstance(lltype, Const) and not isinstance(lltype, GridConstant)
            and _is_grid_constant_shortcut(lltype.inner_type)):
        return Const(GridConstant(lltype.inner_type))
    return lltype


from little_kernel.runtime.abi import register_struct_layout
from little_kernel.backends.nv.common.runtime._tensor_map import CUtensorMap

register_struct_layout(TmaDescriptor, CUtensorMap)
