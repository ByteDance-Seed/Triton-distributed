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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software.
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
"""Explicit-width floating-point literal constructors (ll.f32 / ll.f64).

Why: a bare Python float in kernel source is emitted by the AST codegen as a
bare C++ double literal.  In an expression whose other operands are ``float``
(fp32) the C++ usual arithmetic conversions then promote the whole expression
to ``double``: on sm_90 hardware that manufactures F2F.F64.F32 chains on the
1:64 fp64 pipe and builds an f64-rounded intermediate instead of the declared
fp32 rounding chain (correctness hazard for tolerance-tight kernels, plus a
performance cliff).

These constructors pin the width of a numeric literal without changing any
default behavior:

- ``ll.f32(x)`` — the constant is emitted as an ``f``-suffixed C++ float32
  literal (e.g. ``(1.5f)``).  Non-constant arguments lower to
  ``static_cast<float>(...)``.
- ``ll.f64(x)`` — explicit double literal (documents intended double
  precision; same numeric value as the legacy bare-float emission).

Both forms work inline in a kernel body and at module scope for shared
constants, e.g.::

    LOG2E = ll.f32(1.4426950408889634)

    @lk.ll_kernel(backend="cuda", is_entry=True)
    def k(x: ll.Tensor[ll.float32], ...):
        if x[0] > ll.f32(-87.0):
            x[0] = x[0] * LOG2E

The module-scope form returns a ``float`` subclass so it behaves exactly like
a builtin float for all host-side Python arithmetic; the CUDA codegen
recognizes the marker class when the constant is folded out of the kernel's
global context and emits the suffixed literal.

Legality: the argument must be an int or float (not bool, not str).  This is
the same contract as C++ user literals — a free function with a string or a
non-numeric object would be meaningless as a numeric constant.
"""

import math

from little_kernel.core.type_system import float32, float64
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR


class _F32Literal(float):
    """Marker subclass of ``float``: CUDA codegen emits this constant with an
    ``f`` suffix instead of as a bare C++ double literal.

    Behaves exactly like ``float`` for Python arithmetic (arithmetic results
    are plain ``float``, so the marker never escapes into host-side math); the
    marker is only consulted by CUDA codegen's constant folding.
    """

    __slots__ = ()


class _F64Literal(float):
    """Marker subclass of ``float``: explicit double-precision literal.

    Emitted identically to a legacy bare float literal; exists so frontends
    can document intended double precision and so the opt-in
    ``LK_WARN_FLOAT_LITERAL`` probe can tell pinned-width and legacy literals
    apart.
    """

    __slots__ = ()


def _check_literal_operand(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name}() expects an int or float constant, got {type(x).__name__}")


def f32(x) -> float:
    """Explicit float32 literal constructor (see module docstring)."""
    _check_literal_operand(x, "f32")
    return _F32Literal(x)


def f64(x) -> float:
    """Explicit float64 literal constructor (see module docstring)."""
    _check_literal_operand(x, "f64")
    return _F64Literal(x)


# Type-inference protocol (same EVAL_RETURN_TYPE_ATTR contract the @builtin
# decorator sets): ll.f32(...) has type float32, ll.f64(...) float64, so
# assignments and comparisons containing them infer correctly.
setattr(f32, EVAL_RETURN_TYPE_ATTR, lambda *_, **__: float32)
setattr(f64, EVAL_RETURN_TYPE_ATTR, lambda *_, **__: float64)

# ---------------------------------------------------------------------------
# Codegen-side helpers (shared by expression/call codegen; kept here so the
# formatting rule lives next to the contract that defines it).
# ---------------------------------------------------------------------------


def format_f32_constant(value: float) -> str:
    """Render a Python float as a C++ float32 constant expression."""
    value = float(value)
    if math.isnan(value):
        return "((float)NAN)"
    if math.isinf(value):
        return "((float)INFINITY)" if value > 0 else "(-(float)INFINITY)"
    # repr() round-trips the double; the trailing 'f' makes the C++ literal
    # float32 (round-to-nearest of the same decimal), and the parentheses make
    # it safe inside any binary/unary expression (e.g. negative values).
    return f"({repr(value)}f)"


def format_f64_constant(value: float) -> str:
    """Render a Python float as a C++ double constant expression.

    Matches the legacy bare-literal emission (``str(value)``) numerically,
    parenthesized so it is safe inside any expression.
    """
    value = float(value)
    if math.isnan(value):
        return "(NAN)"
    if math.isinf(value):
        return "(INFINITY)" if value > 0 else "(-INFINITY)"
    return f"({repr(value)})"
