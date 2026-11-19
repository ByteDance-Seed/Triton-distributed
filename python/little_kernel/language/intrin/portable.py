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
"""Portable arithmetic primitives shared by backend language views."""
from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.type_system import int32, LLType


def codegen_cdiv(a, b):
    # Parenthesize substituted operands: a and b may be composite expressions
    # (e.g. "k + 1"), and bare interpolation into "/ b" would rebind precedence.
    return Builtin(body="", includes=[], return_val=f"((({a}) + ({b}) - 1) / ({b}))")


@builtin(
    eval_return_type=lambda a_type, b_type: a_type
    if isinstance(a_type, LLType) else (b_type if isinstance(b_type, LLType) else int32), codegen_func=codegen_cdiv)
def cdiv(a, b):
    """Ceiling division: (a + b - 1) // b
    Returns the same type as the first argument (or second if first is not LLType).
    If both arguments are constants, evaluates immediately.
    """
    # If both arguments are constants (int), evaluate immediately
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    # Otherwise, this should be handled by codegen
    raise RuntimeError("cdiv should not be called in compilation with non-constant arguments")


def codegen_align_power_of_2(a, b):
    # Uses bitwise formula: (a + b - 1) & ~(b - 1). REQUIRES b is a power of 2.
    # Parenthesize substituted operands: a and b may be composite expressions
    # (e.g. "k + 1"), and bare interpolation into "& ~(b - 1)" would rebind
    # precedence.
    return Builtin(body="", includes=[], return_val=f"((({a}) + ({b}) - 1) & ~(({b}) - 1))")


@builtin(
    eval_return_type=lambda a_type, b_type: a_type if isinstance(a_type, LLType) else
    (b_type if isinstance(b_type, LLType) else int32), codegen_func=codegen_align_power_of_2)
def align_power_of_2(a, b):
    """Align a up to the given byte alignment (power-of-2 only).

    REQUIRES b is a power of 2 (e.g. 2, 4, 8, 16, 32). The implementation uses
    (a + b - 1) & ~(b - 1), which is only correct when b is a power of 2.
    Using non-power-of-2 values (e.g. 3) will produce incorrect results.

    Returns the same type as the first argument (or second if first is not LLType).
    If both arguments are constants, evaluates immediately.
    """
    if isinstance(a, int) and isinstance(b, int):
        if b <= 0 or (b & (b - 1)) != 0:
            raise ValueError(f"align_power_of_2: alignment {b} must be a positive power of 2")
        return (a + b - 1) & ~(b - 1)
    # Otherwise, this should be handled by codegen
    raise RuntimeError("align_power_of_2 should not be called in compilation with non-constant arguments")


def codegen_min_val(a, b):
    return Builtin(body="", includes=[], return_val=f"min({a}, {b})")


@builtin(eval_return_type=lambda a_type, b_type: a_type, codegen_func=codegen_min_val)
def min_val(a, b):
    """CUDA min function."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    raise RuntimeError("should not be called in compilation")


def codegen_max_val(a, b):
    # Use max for general case (works for both int and float)
    return Builtin(body="", includes=[], return_val=f"max({a}, {b})")


@builtin(eval_return_type=lambda a_type, b_type: a_type, codegen_func=codegen_max_val)
def max_val(a, b):
    """CUDA max function."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    raise RuntimeError("should not be called in compilation")


def codegen_select(cond, on_true, on_false):
    return Builtin(body="", includes=[], return_val=f"(({cond}) ? ({on_true}) : ({on_false}))")


@builtin(eval_return_type=lambda cond_type, a_type, b_type: a_type, codegen_func=codegen_select)
def select(cond, on_true, on_false):
    """Select ``on_true`` when ``cond`` is nonzero, else ``on_false``.

    Emits the C ternary on scalar lanes; the result type is the type of
    ``on_true`` (any scalar dtype: fp32/fp16/bf16/int/uint).  Both arms pass
    through unmodified — the ternary introduces no FP operation of its own,
    so the selection itself is immune to the fixed ``--use_fast_math`` task
    build's rewrites (pair with the ``_rn`` arithmetic family when the arm
    expressions must also stay RN-exact).  Intended for per-lane masked
    arithmetic after typed vector loads (``gmem_ld_v4f32_nc`` +
    ``gmem_ld_v4s32_nc`` lanes), removing the need for a per-frontend Builtin
    body string.  For bit-level 32-bit lane selects see ``selp_b32``.
    Constants fold at compile time.
    """
    if isinstance(cond, int):
        return on_true if cond else on_false
    raise RuntimeError("should not be called in compilation")
