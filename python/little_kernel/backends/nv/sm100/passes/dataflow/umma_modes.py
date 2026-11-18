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
"""Normalize UMMA effects for analysis, then coalesce them for codegen."""

from __future__ import annotations

import ast

from little_kernel.core.memory_effects import Effect, W_TMEM_TCGEN05_COMMIT
from little_kernel.language.builtin_base import MEMORY_EFFECTS_ATTR
from little_kernel.backends.nv.sm100.passes.dataflow.compile_time_unroll import _const_int

_CASTS = {"uint32", "uint64", "int32", "int64"}
NORMALIZED_EFFECTS_ATTR = "_lk_normalized_memory_effects"
_KBLOCK_ARITIES = {
    "umma_f16_ss_1sm_kblock": 4,
    "umma_f16_ts_1sm_kblock": 4,
    "umma_f16_ts_2sm_kblock": 4,
    "umma_f16_ss_2sm_kblock": 4,
}
_KBLOCK_MODES = set(_KBLOCK_ARITIES)
_LOW_LEVEL_UMMA = {
    "umma_f16_ss_1sm",
    "umma_f16_ts_1sm",
    "umma_f16_ts_2sm",
    "umma_f16_ss_2sm",
    "umma_f16_ss_1sm_kblock",
    "umma_f16_ts_1sm_kblock",
    "umma_f16_ss_2sm_kblock",
}


def _call_name(node: ast.Call) -> tuple[str | None, str | None]:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id, func.attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None, None


def _is_call(node: ast.AST, owner: str, name: str) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) == (owner, name)


def _key(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _same(a: ast.AST, b: ast.AST) -> bool:
    return _key(a) == _key(b)


def _clone(expr: ast.expr) -> ast.expr:
    """Copy syntax without compiler-added attributes such as inferred LL types."""
    return ast.parse(ast.unparse(expr), mode="eval").body


def _unwrap(expr: ast.expr) -> ast.expr:
    while isinstance(expr, ast.Call) and _call_name(expr)[1] in _CASTS and len(expr.args) == 1:
        expr = expr.args[0]
    return expr


def _const_mode(expr: ast.expr) -> int | None:
    expr = _unwrap(expr)
    if isinstance(expr, ast.Constant) and expr.value in {0, 1, False, True}:
        return int(bool(expr.value))
    return None


def _mark_kblock_effects(call: ast.Call, name: str, zero_init: bool) -> ast.Call:
    from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

    effects = getattr(getattr(ri, name), MEMORY_EFFECTS_ATTR)
    if zero_init:
        effects = tuple(
            Effect(W_TMEM_TCGEN05_COMMIT, item.operand, item.region) if item.operand in {"d_mem", "tmem_d"} else item
            for item in effects)
    setattr(call, NORMALIZED_EFFECTS_ATTR, effects)
    call.keywords.append(ast.keyword(arg="zero_init", value=ast.Constant(value=zero_init)))
    return call


def _assigned_value(stmt: ast.stmt, name: str) -> ast.expr | None:
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
        return stmt.value
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name):
        return stmt.value
    return None


class _AstIndex:

    def __init__(self, tree: ast.AST) -> None:
        self.parents: dict[int, ast.AST] = {}
        self.positions: dict[int, tuple[list[ast.stmt], int]] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[id(child)] = parent
            for _field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, ast.stmt):
                            self.positions[id(item)] = (value, index)

    def ancestors(self, node: ast.AST):
        while (node := self.parents.get(id(node))) is not None:
            yield node

    def local_definition(self, call: ast.Call, name: str) -> ast.expr | None:
        stmt = next((node for node in self.ancestors(call) if isinstance(node, ast.stmt)), None)
        if stmt is None or id(stmt) not in self.positions:
            return None
        body, index = self.positions[id(stmt)]
        for previous in reversed(body[:index]):
            if (value := _assigned_value(previous, name)) is not None:
                return value
        return None

    def loop_for(self, call: ast.Call, name: str) -> ast.While | ast.For | None:
        for node in self.ancestors(call):
            if isinstance(node, ast.While):
                if _while_var(node) == name or any(_assigned_value(stmt, name) is not None for stmt in node.body):
                    return node
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == name:
                return node
        return None


def _while_var(node: ast.While) -> str | None:
    test = node.test
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Lt) and len(test.comparators) == 1):
        return test.left.id
    return None


def _simple_increment(stmt: ast.stmt, name: str) -> bool:
    value = _assigned_value(stmt, name)
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return False
    return (isinstance(value.left, ast.Name) and value.left.id == name and _const_mode(value.right) == 1)


def _range_start(node: ast.For) -> ast.expr | None:
    call = node.iter
    if isinstance(call, ast.Call) and _call_name(call)[1] == "unroll" and len(call.args) == 1:
        call = call.args[0]
    if not isinstance(call, ast.Call) or _call_name(call)[1] != "range" or not call.args:
        return None
    return ast.Constant(value=0) if len(call.args) == 1 else call.args[0]


def _predicate_parts(expr: ast.expr) -> list[tuple[str, ast.expr]] | None:
    expr = _unwrap(expr)
    values = expr.values if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or) else [expr]
    if not 1 <= len(values) <= 2:
        return None
    out: list[tuple[str, ast.expr]] = []
    for value in values:
        value = _unwrap(value)
        if (not isinstance(value, ast.Compare) or not isinstance(value.left, ast.Name) or len(value.ops) != 1
                or not isinstance(value.ops[0], (ast.NotEq, ast.Gt)) or len(value.comparators) != 1):
            return None
        out.append((value.left.id, _unwrap(value.comparators[0])))
    return out


def _prove_first_point(
    call: ast.Call,
    accum: ast.expr,
    index: _AstIndex,
    *,
    require_outer_while: bool = True,
) -> None:
    proof = accum
    if isinstance(_unwrap(proof), ast.Name):
        name = _unwrap(proof).id
        proof = index.local_definition(call, name) or proof
    parts = _predicate_parts(proof)
    if parts is None:
        raise ValueError("dynamic UMMA accum must be `outer != start` or `outer != start or inner != start`")

    while_count = 0
    for name, start in parts:
        loop = index.loop_for(call, name)
        if isinstance(loop, ast.While):
            while_count += 1
            updates = [stmt for stmt in loop.body if _assigned_value(stmt, name) is not None]
            if loop.orelse or len(updates) != 1 or not _simple_increment(updates[0], name):
                raise ValueError(f"UMMA reduction loop {name!r} must update once with `{name} = {name} + 1`")
            if any(isinstance(node, (ast.Break, ast.Continue, ast.Return)) for node in ast.walk(loop)):
                raise ValueError(f"UMMA reduction loop {name!r} cannot contain early exits")
            body, position = index.positions[id(loop)]
            init = next(
                (value for stmt in reversed(body[:position]) if (value := _assigned_value(stmt, name)) is not None),
                None,
            )
            if init is None or not _same(_unwrap(init), start):
                raise ValueError(f"UMMA reduction loop {name!r} start does not match its accum predicate")
        elif isinstance(loop, ast.For):
            loop_start = _range_start(loop)
            if loop_start is None or not _same(_unwrap(loop_start), start):
                raise ValueError(f"UMMA inner loop {name!r} start does not match its accum predicate")
        else:
            raise ValueError(f"UMMA accum variable {name!r} is not an enclosing canonical loop")
    if require_outer_while and while_count != 1:
        raise ValueError("dynamic UMMA accum must describe exactly one outer while reduction")
    if not require_outer_while and while_count > 1:
        raise ValueError("dynamic UMMA mode can describe at most one outer while reduction")


class _UmmaEffectNormalizer(ast.NodeTransformer):

    def __init__(self, tree: ast.AST, const_env: dict) -> None:
        self.index = _AstIndex(tree)
        self.const_env = const_env

    def visit_Expr(self, node: ast.Expr):
        node = self.generic_visit(node)
        call = node.value
        if isinstance(call, ast.Call) and _call_name(call)[0] == "ri" and _call_name(call)[1] in _KBLOCK_MODES:
            name = _call_name(call)[1]
            arity = _KBLOCK_ARITIES[name]
            zero_keywords = [keyword for keyword in call.keywords if keyword.arg == "zero_init"]
            if len(zero_keywords) > 1:
                raise ValueError(f"{name} received zero_init more than once")
            if zero_keywords:
                zero_init = zero_keywords[0].value
            elif len(call.args) == arity:
                zero_init = ast.Constant(value=False)
            else:
                raise ValueError(f"{name} expects {arity} operands plus keyword-only zero_init")

            k = _const_int(call.args[arity - 1], self.const_env)
            if k is None:
                raise ValueError(f"{name} requires a compile-time logical K extent")
            if k <= 0 or k % 16:
                raise ValueError(f"{name} logical K extent must be a positive multiple of 16, got {k}")

            stripped = _clone(call)
            assert isinstance(stripped, ast.Call)
            stripped.keywords = [keyword for keyword in stripped.keywords if keyword.arg != "zero_init"]
            mode = _const_mode(zero_init)
            if mode is not None:
                _mark_kblock_effects(stripped, name, bool(mode))
                return ast.copy_location(ast.Expr(value=stripped), node)

            proof = _unwrap(zero_init)
            if isinstance(proof, ast.Name):
                proof = self.index.local_definition(call, proof.id) or proof
            _prove_first_point(
                call,
                _negate(_unwrap(proof)),
                self.index,
                require_outer_while=False,
            )
            init_call = _mark_kblock_effects(_clone(stripped), name, True)
            accum_call = _mark_kblock_effects(_clone(stripped), name, False)
            return ast.copy_location(
                ast.If(
                    test=_clone(_unwrap(zero_init)),
                    body=[ast.copy_location(ast.Expr(value=init_call), node)],
                    orelse=[ast.copy_location(ast.Expr(value=accum_call), node)],
                ),
                node,
            )

        if not _is_call(call, "ri", "umma_f16_ss_2sm") or len(call.args) != 5:
            return node
        mode = _const_mode(call.args[-1])
        if mode is not None:
            call.func.attr = "umma_f16_ss_2sm_accum" if mode else "umma_f16_ss_2sm_init"
            call.args = call.args[:-1]
            return node

        _prove_first_point(call, call.args[-1], self.index)
        init_call = _clone(call)
        init_call.func.attr = "umma_f16_ss_2sm_init"
        init_call.args = init_call.args[:-1]
        accum_call = _clone(call)
        accum_call.func.attr = "umma_f16_ss_2sm_accum"
        accum_call.args = accum_call.args[:-1]
        test = ast.Compare(left=_clone(_unwrap(call.args[-1])), ops=[ast.Eq()], comparators=[ast.Constant(value=0)])
        return ast.copy_location(
            ast.If(
                test=test,
                body=[ast.copy_location(ast.Expr(value=init_call), node)],
                orelse=[ast.copy_location(ast.Expr(value=accum_call), node)],
            ),
            node,
        )


def _reject_generic_umma(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node)[0] != "ri":
            continue
        name = _call_name(node)[1]
        if name == "umma_f16_ss_2sm":
            raise ValueError(f"generic ri.{name} must be normalized before dataflow analysis")
        if name in _KBLOCK_MODES and not hasattr(node, NORMALIZED_EFFECTS_ATTR):
            raise ValueError(f"generic ri.{name} must have normalized effects before dataflow analysis")


def umma_effect_normalization_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    _UmmaEffectNormalizer(tree, ctx or {}).visit(tree)
    _reject_generic_umma(tree)
    ast.fix_missing_locations(tree)
    return tree


def _negate(expr: ast.expr) -> ast.expr:
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return expr.operand
    if isinstance(expr, ast.BoolOp):
        op = ast.Or() if isinstance(expr.op, ast.And) else ast.And()
        return ast.BoolOp(op=op, values=[_negate(value) for value in expr.values])
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        if (isinstance(expr.comparators[0], ast.Constant) and expr.comparators[0].value == 0
                and isinstance(expr.left, (ast.BoolOp, ast.Compare, ast.UnaryOp))):
            if isinstance(expr.ops[0], ast.Eq):
                return expr.left
            if isinstance(expr.ops[0], ast.NotEq):
                return _negate(expr.left)
        inverse = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In,
        }.get(type(expr.ops[0]))
        if inverse is not None:
            return ast.Compare(left=expr.left, ops=[inverse()], comparators=expr.comparators)
    return ast.UnaryOp(op=ast.Not(), operand=expr)


def _low_level_umma(stmt: ast.stmt) -> tuple[ast.Call, int] | None:
    if (not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call) or _call_name(stmt.value)[0] != "ll"
            or _call_name(stmt.value)[1] not in _LOW_LEVEL_UMMA):
        return None
    call = stmt.value
    if (mode := _const_mode(call.args[-1])) is None:
        return None
    return call, mode


class _UmmaCodegenCoalescer(ast.NodeTransformer):

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        if len(node.body) != 1 or len(node.orelse) != 1:
            return node
        lhs = _low_level_umma(node.body[0])
        rhs = _low_level_umma(node.orelse[0])
        if lhs is None or rhs is None or lhs[1] == rhs[1]:
            return node
        if _call_name(lhs[0]) != _call_name(rhs[0]):
            return node
        if any(not _same(a, b) for a, b in zip(lhs[0].args[:-1], rhs[0].args[:-1])):
            return node
        accum = _clone(node.test if lhs[1] == 1 else _negate(node.test))
        call = lhs[0]
        call.args[-1] = ast.Call(
            func=ast.Attribute(value=ast.Name(id="ll", ctx=ast.Load()), attr="uint32", ctx=ast.Load()),
            args=[accum],
            keywords=[],
        )
        return ast.copy_location(ast.Expr(value=call), node)


def umma_codegen_coalescing_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    del ctx
    _UmmaCodegenCoalescer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree
