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
"""
Atomic operation intrinsics.
"""

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import int32, LLType

# ==============================================================================
# Helpers for dynamic return type inference
# ==============================================================================


def _atomic_return_type_from_addr(addr: LLType) -> LLType:
    """Infer return type from pointer/tensor addr. Atomic ops return old value (same type as element)."""
    if addr is None:
        return int32
    if hasattr(addr, "is_pointer") and addr.is_pointer() and hasattr(addr, "inner_type"):
        return addr.inner_type
    if hasattr(addr, "is_tensor") and addr.is_tensor() and hasattr(addr, "element_type"):
        return addr.element_type
    return int32  # Fallback when type cannot be inferred


# ==============================================================================
# atomicAdd
# ==============================================================================


def codegen_atomic_add(addr, val):
    return Builtin(body="", includes=[], return_val=f"atomicAdd({addr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_add)
def atomic_add(addr, val):
    """atomicAdd on a pointer. Returns old value."""
    raise RuntimeError("should not call atomic_add in compilation")


# ==============================================================================
# atomicCAS_system (cross-rank / system-wide)
# ==============================================================================


def codegen_atomic_cas_system(addr, compare, val):
    return Builtin(body="", includes=[], return_val=f"atomicCAS_system({addr}, {compare}, {val})")


@builtin(eval_return_type=lambda addr, compare, val: _atomic_return_type_from_addr(addr),
         codegen_func=codegen_atomic_cas_system)
def atomic_cas_system(addr, compare, val):
    """System-scope atomicCAS. Returns old value."""
    raise RuntimeError("should not call atomic_cas_system in compilation")


# ==============================================================================
# atomicCAS (device-scope)
# ==============================================================================


def codegen_atomic_cas(addr, compare, val):
    return Builtin(body="", includes=[], return_val=f"atomicCAS({addr}, {compare}, {val})")


@builtin(eval_return_type=lambda addr, compare, val: _atomic_return_type_from_addr(addr),
         codegen_func=codegen_atomic_cas)
def atomic_cas(addr, compare, val):
    """Device-scope atomicCAS. Returns old value."""
    raise RuntimeError("should not call atomic_cas in compilation")


# ==============================================================================
# atomicMin
# ==============================================================================


def codegen_atomic_min(addr, val):
    return Builtin(body="", includes=[], return_val=f"atomicMin({addr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_min)
def atomic_min(addr, val):
    """atomicMin on a pointer. Returns old value."""
    raise RuntimeError("should not call atomic_min in compilation")


# ==============================================================================
# atomicMax
# ==============================================================================


def codegen_atomic_max(addr, val):
    return Builtin(body="", includes=[], return_val=f"atomicMax({addr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_max)
def atomic_max(addr, val):
    """atomicMax on a pointer. Returns old value."""
    raise RuntimeError("should not call atomic_max in compilation")


# ==============================================================================
# atomicOr
# ==============================================================================


def codegen_atomic_or(addr, val):
    return Builtin(body="", includes=[], return_val=f"atomicOr({addr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_or)
def atomic_or(addr, val):
    """atomicOr on a pointer. Returns old value."""
    raise RuntimeError("should not call atomic_or in compilation")


# ==============================================================================
# atomicAnd
# ==============================================================================


def codegen_atomic_and(addr, val):
    return Builtin(body="", includes=[], return_val=f"atomicAnd({addr}, {val})")


@builtin(eval_return_type=lambda addr, val: _atomic_return_type_from_addr(addr), codegen_func=codegen_atomic_and)
def atomic_and(addr, val):
    """atomicAnd on a pointer. Returns old value."""
    raise RuntimeError("should not call atomic_and in compilation")
