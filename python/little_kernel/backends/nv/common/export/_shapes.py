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
"""Host-side shape expressions.

A tiny arithmetic language (no eval/exec) translated to the C++ text emitted
in the host binding/launcher. It is the single owner of dimension subscripts,
strides, scalar names, list aggregates (sum/len/has), optional-input
``present()`` and the ``device_sms()`` launch builtin. Derived constexpr
selectors validate against the same language before becoming Model.__init__
expressions.
"""

from __future__ import annotations

import ast
import re


def _flattened_leading_tma_sources(tensor: str, tensor_rank: int) -> dict[str, str]:
    """Return the canonical row-major [leading product, last dim] 2-D view."""
    last = tensor_rank - 1
    return {
        "inner": f"{tensor}[{last}]",
        "outer": "*".join(f"{tensor}[{index}]" for index in range(last)),
        "outer_stride": f"{tensor}[{last}]",
    }


def _shape_expr(source: object, ranks: dict[str, int], scalars: set[str] | frozenset[str] = frozenset(), *,
                allow_float: bool = False, optional_inputs: set[str] | frozenset[str] = frozenset(),
                lists: dict[str, bool] | None = None, derived: dict[str, str] | None = None) -> str:
    """Translate a tiny shape arithmetic language to C++, without eval/exec.

    ``derived`` maps shape-legal derived integer scalar names to their already
    emitted C++ expressions: a Name reference emits the parenthesized inline
    substitution instead of the bare variable. This lets allocations that
    precede the scalar's ``const int64_t`` definition in the generated binding
    (output/workspace ``at::empty`` sites) still size themselves by a derived
    extent that launch.grid and kernel args share."""
    if type(source) is int:
        if source < 0:
            raise ValueError("shape constants must be nonnegative")
        return str(source)
    if not isinstance(source, str):
        raise ValueError("shape expression must be a string or nonnegative integer")
    node = ast.parse(source, mode="eval").body

    def emit(expr: ast.expr, *, _root: bool = False) -> str:
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "present"):
            # present(x) is 1 when the optional model input x carried a real
            # tensor at the ABI boundary, 0 when the caller passed None (the
            # binding substitutes a zero-sized tensor). Kernels branch on a
            # declared scalar computed from it instead of dereferencing x.
            if len(expr.args) != 1 or expr.keywords or not isinstance(expr.args[0], ast.Name):
                raise ValueError("present requires one input name argument")
            target = expr.args[0].id
            if target not in optional_inputs:
                raise ValueError(f"present() must name an input declared optional: {target}")
            return f"({target}_opt.has_value() ? 1 : 0)"
        if isinstance(expr, ast.Constant) and type(expr.value) is int and expr.value >= 0:
            return str(expr.value)
        if allow_float and isinstance(expr, ast.Constant) and type(expr.value) is float and expr.value >= 0:
            return repr(expr.value)
        if isinstance(expr, ast.Name) and expr.id in scalars:
            return expr.id
        if isinstance(expr, ast.Name) and derived is not None and expr.id in derived:
            # Inline substitution of a validated derived integer scalar: the
            # generated binding defines the scalar variable only after the
            # output/workspace at::empty sites, so allocations sized by a
            # derived extent expand the computed expression here instead of
            # referencing the not-yet-declared variable. The stored spelling
            # is already fully expanded (nested derived references were
            # substituted when the entry was recorded); parenthesize so any
            # compound expression binds correctly under the caller's
            # operators. Names in `scalars` win so definition-site emission
            # (the scalar's own const int64_t line, launch expressions
            # emitted after it) keeps the readable variable reference.
            return f"({derived[expr.id]})"
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("sum", "len", "has")):
            if (lists is None or len(expr.args) != 1 or expr.keywords or not isinstance(expr.args[0], ast.Name)
                    or expr.args[0].id not in lists):
                raise ValueError(f"{expr.func.id} requires one declared host int-list argument")
            name = expr.args[0].id
            if expr.func.id == "has":
                return f"({name}.has_value() ? 1 : 0)" if lists[name] else "1"
            helper = "lk_host_sum" if expr.func.id == "sum" else "lk_host_len"
            if lists[name]:
                return f"({name}.has_value() ? {helper}(*{name}) : 0)"
            return f"{helper}({name})"
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
            name = expr.value.id
            index = expr.slice
            if name not in ranks or not isinstance(index, ast.Constant) or type(index.value) is not int:
                raise ValueError(f"invalid tensor dimension in {source!r}")
            if not 0 <= index.value < ranks[name]:
                raise ValueError(f"dimension out of range in {source!r}")
            return f"{name}.size({index.value})"
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "stride"
                and isinstance(expr.func.value, ast.Name)):
            name = expr.func.value.id
            if (name not in ranks or len(expr.args) != 1 or expr.keywords or not isinstance(expr.args[0], ast.Constant)
                    or type(expr.args[0].value) is not int or not 0 <= expr.args[0].value < ranks[name]):
                raise ValueError(f"invalid tensor stride dimension in {source!r}")
            return f"{name}.stride({expr.args[0].value})"
        if isinstance(expr, ast.BinOp) and type(expr.op) in (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod):
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "/", ast.Mod: "%"}[type(expr.op)]
            if isinstance(expr.op, ast.Mod):
                # % shares the integer-typing rule of // with one soundness
                # gap closed: unlike C / (which promotes to double), C % has
                # no floating-point form, so any operand that can be a float
                # (float literal under allow_float, or a Div/sqrt subtree)
                # would emit uncompilable C++. Reject those subtrees here.
                def _may_be_float(node: ast.expr) -> bool:
                    return ((isinstance(node, ast.Constant) and type(node.value) is float)
                            or (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                                and _may_be_float(node.operand))
                            or (isinstance(node, ast.BinOp) and
                                (isinstance(node.op, ast.Div) or _may_be_float(node.left) or _may_be_float(node.right)))
                            or
                            (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt"))

                if _may_be_float(expr.left) or _may_be_float(expr.right):
                    raise ValueError(f"% requires integer operands in {source!r}")
            return f"({emit(expr.left)} {op} {emit(expr.right)})"
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
            # Unary minus/plus on any admissible subexpression: numeric
            # literals ("-1.5" under allow_float), declared scalars, tensor
            # dimensions or compound expressions. The recursive emit reuses
            # the same allow_float/scalar/rank resolution as every other
            # branch, so authors need not hand-expand "0 - v". A signed
            # *integer literal* is only admissible as the whole expression
            # (the top-level "-3" form): nested below a call/operator (e.g.
            # ceil_div(x[0], -8) or max(..., -1)) it is still a negative
            # shape size and keeps the historical rejection. (A bare
            # negative JSON int is likewise rejected by the int entry rule.)
            if (not _root and isinstance(expr.operand, ast.Constant) and type(expr.operand.value) is int):
                raise ValueError(f"unsupported shape expression: {source!r}")
            sign = "-" if isinstance(expr.op, ast.USub) else "+"
            return f"({sign}({emit(expr.operand)}))"
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or not isinstance(expr.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                raise ValueError(f"only single-op < <= > >= == != comparisons are supported in {source!r}")
            op = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq:
                  "!="}[type(expr.ops[0])]
            return f"((int64_t)({emit(expr.left)} {op} {emit(expr.comparators[0])}))"
        if allow_float and isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            return f"({emit(expr.left)} / {emit(expr.right)})"
        if allow_float and isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "sqrt":
            if len(expr.args) != 1 or expr.keywords:
                raise ValueError("sqrt requires one positional argument")
            return f"std::sqrt({emit(expr.args[0])})"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("min", "max"):
            if len(expr.args) != 2 or expr.keywords:
                raise ValueError(f"{expr.func.id} requires two positional arguments")
            return f"lk_{expr.func.id}({emit(expr.args[0])}, {emit(expr.args[1])})"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "device_sms":
            # Device SM-count builtin: resolved on the host at launch time via
            # cudaDeviceGetAttribute. This is what lets a persistent one-wave
            # cluster grid (e.g. a 2-CTA producer/consumer cluster sized as
            # min(device_sms() // 2, tiles) * 2) observe the device instead of
            # fixing an oversized grid that strands the ragged tail on the
            # cluster barrier. The emitted expression stays an ordinary int64
            # size, so the existing positive/cluster-divisibility checks apply.
            if expr.args or expr.keywords:
                raise ValueError("device_sms takes no arguments")
            return "lk_device_sms()"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "select":
            if len(expr.args) != 3 or expr.keywords:
                raise ValueError("select requires exactly three positional arguments")
            return f"(({emit(expr.args[0])}) ? ({emit(expr.args[1])}) : ({emit(expr.args[2])}))"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "ceil_div":
            if len(expr.args) != 2 or expr.keywords:
                raise ValueError("ceil_div requires exactly two positional arguments")
            return f"lk_ceil_div({emit(expr.args[0])}, {emit(expr.args[1])})"
        raise ValueError(f"unsupported shape expression: {source!r}")

    return emit(node, _root=True)


_DERIVED_EXPR_NODES = (
    ast.Expression,
    ast.Compare,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Load,
)


def _derived_constexpr_expr(source: object, preceding: frozenset[str]) -> str:
    """Validate a derived-constexpr expression and return its C++ spelling."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("derived constexpr expression must be a nonempty string")
    tree = ast.parse(source, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _DERIVED_EXPR_NODES):
            raise ValueError(f"unsupported derived constexpr expression: {source!r}")
        if isinstance(node, ast.Name) and node.id not in preceding:
            raise ValueError(f"derived constexpr expression may only reference preceding "
                             f"constructor arguments: {source!r}")
        if isinstance(node, ast.Compare) and (len(node.ops) != 1 or len(node.comparators) != 1):
            raise ValueError(f"derived constexpr comparison needs exactly one operator: {source!r}")
    # Reuse the host shape-expression machinery as a second, language-level
    # validation: ranks/lists are empty here, so only constants, the preceding
    # constructor names and arithmetic/comparison operators survive
    # translation. Float operands are admitted so selectors over float
    # constructor args (momentum != 0.0) render as valid host C++.
    cpp = _shape_expr(source, {}, preceding, allow_float=True)
    return cpp


_CUDA_PRELUDE_LINE = re.compile(r"[ \t]*(?:#|//)[^\n]*\n|[ \t]*\n")


def _split_cuda_prelude(source: str) -> tuple[str, str]:
    """Split a codegen CUDA unit into its global-scope prelude and its body.

    The prelude is the run of leading preprocessor directives, comment lines
    and blank lines (typically the generated ``#include`` block). Variant
    fanout wraps each specialization in its own namespace, and an #include
    pulled into that namespace would mis-scope the headers, so the prelude is
    emitted once at global scope.
    """
    offset = 0
    while True:
        match = _CUDA_PRELUDE_LINE.match(source, offset)
        if match is None:
            break
        offset = match.end()
    return source[:offset], source[offset:]
