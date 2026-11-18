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
"""Provably-static integer local analysis for code generation.

A frontend commonly pins workload constants as typed integer locals
initialized from literals or arithmetic over other such locals::

    MAX_N: ll.int32 = 52
    ll.alloc_shared_memory("off_tbl", ll.int32, MAX_N)
    ll.alloc_shared_memory("tile_sm", ll.bfloat16, STAGES * TILE * HEAD_DIM)

CUDA requires the extent of a ``__shared__`` array to be an integer
constant expression; an ordinary ``int32_t MAX_N = 52;`` local is not one,
so the generated kernel is a variable-length array with static storage
duration and nvcc rejects it.  The same applies to local C arrays, whose
bounds must be constant expressions in C++.

This module implements the shared analysis that proves which integer
locals are *compile-time constants*:

* exactly one definition, a top-level function statement of the form
  ``X = <expr>`` or ``X: <int-annotation> = <expr>``;
* ``<expr>`` evaluates to a compile-time integer using only integer
  literals, other earlier-defined static names, and the arithmetic/
  bitwise operators supported by the source language's constant
  semantics (``+ - * // % << >> & | ^ ~`` and unary ``+ - ~``), with
  C-style truncating division/modulo so ``//``/``%`` constancy checks
  (division by zero) match what nvcc will later evaluate;
* the value fits the declared/inferred integer type's range;
* the name is never rebound or mutated anywhere in the function
  (assignment, annotated assignment, augmented assignment, tuple-unpack
  target, or for-loop target), and is not a function parameter.

A name that fails any check is simply absent from the result: codegen
then keeps the historical plain-local emission, which is always a
conservative fallback (a backend requiring constant extents diagnoses
the resulting program honestly instead of mis-annotating it).
"""

import ast
from typing import Dict, Optional

from little_kernel.core.type_system import IntType, LLType

# Names that must never be treated as static constants even when they
# syntactically qualify (CUDA grid/thread builtins read per-thread state).
_BANNED_NAME_PREFIXES = ("blockIdx", "threadIdx", "blockDim", "gridDim", "warpSize")


def _int_range(ll_type: IntType) -> Optional[tuple]:
    """Inclusive (lo, hi) value range of an integer LLType."""
    bits = getattr(ll_type, "bits", None)
    if not isinstance(bits, int) or bits < 1 or bits > 128:
        return None
    if getattr(ll_type, "special", None) in ("bool", "binary"):
        return None
    if ll_type.signed:
        return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return (0, (1 << bits) - 1)


def _c_div(a: int, b: int) -> int:
    """C-style truncating integer division (Python ``//`` floors)."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def _c_mod(a: int, b: int) -> int:
    """C-style remainder (sign follows the dividend)."""
    return a - _c_div(a, b) * b


class _ConstEvalError(Exception):
    """Internal: expression is not a supported compile-time constant."""


def _eval_const(node: ast.AST, bindings: Dict[str, int]) -> int:
    """Evaluate ``node`` as a compile-time integer constant.

    ``bindings`` maps previously proven static names to their values.
    Raises _ConstEvalError for anything outside the supported fragment:
    integer literals, bound names, binary ``+ - * // % << >> & | ^``,
    and unary ``+ - ~``.  Division/modulo use C truncation semantics and
    reject a zero divisor (mirroring C++ constant-expression rules, where
    a zero division makes the constant expression ill-formed).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _ConstEvalError("non-integer literal")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in bindings:
            return bindings[node.id]
        raise _ConstEvalError(f"unbound name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        v = _eval_const(node.operand, bindings)
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.USub):
            return -v
        return ~v
    if isinstance(node, ast.BinOp):
        a = _eval_const(node.left, bindings)
        b = _eval_const(node.right, bindings)
        op = node.op
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, (ast.FloorDiv, ast.Div)):
            if b == 0:
                raise _ConstEvalError("division by zero")
            return _c_div(a, b)
        if isinstance(op, ast.Mod):
            if b == 0:
                raise _ConstEvalError("modulo by zero")
            return _c_mod(a, b)
        if isinstance(op, ast.LShift):
            if b < 0:
                raise _ConstEvalError("negative shift")
            return a << b
        if isinstance(op, ast.RShift):
            if b < 0:
                raise _ConstEvalError("negative shift")
            return a >> b
        if isinstance(op, ast.BitAnd):
            return a & b
        if isinstance(op, ast.BitOr):
            return a | b
        if isinstance(op, ast.BitXor):
            return a ^ b
        raise _ConstEvalError(f"unsupported operator {type(op).__name__}")
    raise _ConstEvalError(f"unsupported expression {type(node).__name__}")


def try_eval_const_int(node: ast.AST, bindings: Dict[str, int]) -> Optional[int]:
    """Public fold helper: value of ``node`` if it is a compile-time
    integer constant under ``bindings``, else None."""
    try:
        return _eval_const(node, bindings)
    except _ConstEvalError:
        return None


# Backends that take a static array extent (``alloc_shared_memory``,
# ``alloc_local_memory`` and friends) accept that extent spelled either as
# a bare integer expression (``512``) or as a one-element shape container
# (``[512]`` / ``(512,)``) — the latter is the conventional "shape" form
# frontend code uses even for a 1-D scratch buffer.  A braced ``{512}``
# aggregate-init rendering is not valid C++ inside a declarator bound, so
# a single-element container must be treated as the wrapped expression for
# both constness validation and emission.  Only a singleton container is
# admissible: a multi-element list is a shape, not a scalar extent, and
# must keep failing the constant-extent check.
def unwrap_scalar_extent_container(node: ast.AST) -> ast.AST:
    """Return the sole element of a one-element ``ast.List``/``ast.Tuple``
    literal (recursively), otherwise the node itself.

    This normalises the two source-level spellings of a scalar extent so
    later folds and emission render the inner integer expression instead
    of a braced aggregate.
    """
    while isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 1:
        node = node.elts[0]
    return node


def try_eval_const_scalar_int(node: ast.AST, bindings: Dict[str, int]) -> Optional[int]:
    """Like :func:`try_eval_const_int`, but first unwraps a one-element
    list/tuple container so ``[512]`` and ``(512,)`` fold to ``512``."""
    return try_eval_const_int(unwrap_scalar_extent_container(node), bindings)


def _assigned_names(node: ast.AST):
    """Yield every Name bound by an assignment/loop-target subtree."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _assigned_names(elt)
    # Subscript/Attribute stores mutate through the base object; they do
    # not *rebind* the base name, so they are not collected here.  A
    # static integer name is a scalar, so a subscript store through it
    # would already be a type error elsewhere.


def find_mutated_names(func_node: ast.FunctionDef) -> set:
    """Names with more than one binding site, or any non-plain binding.

    A provably-static local has exactly one definition: a top-level
    ``x = <expr>`` / ``x: T = <expr>`` statement.  That single defining
    occurrence is not a mutation.  Everything else is disqualifying:

    * a second binding anywhere (reassignment, tuple re-unpack);
    * an augmented assignment (``x += 1``), which is a read-modify-write;
    * a for-loop target (rebound each iteration);
    * any binding nested inside a control-flow block (the analysis only
      tracks straight-line top-level definitions, and a conditional first
      binding cannot be proven to dominate its uses).

    Conservative: any disqualifying store drops the name from the static
    set, so codegen keeps plain-local emission and the downstream C++
    compiler never sees a mis-annotated ``constexpr``.
    """
    counts: Dict[str, int] = {}
    bad: set = set()

    def record(target):
        for name in _assigned_names(target):
            counts[name] = counts.get(name, 0) + 1

    def record_bad(target):
        names = set(_assigned_names(target))
        bad.update(names)
        for n in names:
            counts[n] = counts.get(n, 0) + 1

    def walk_body(body, top_level: bool):
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                if top_level:
                    for tgt in stmt.targets:
                        record(tgt)
                else:
                    for tgt in stmt.targets:
                        record_bad(tgt)
                # Nested payloads (e.g. subscript slices) contain no
                # rebindings of scalar names; only control-flow structure
                # introduces new scopes to scan.
            elif isinstance(stmt, ast.AnnAssign):
                (record if top_level else record_bad)(stmt.target)
            elif isinstance(stmt, ast.AugAssign):
                record_bad(stmt.target)
            elif isinstance(stmt, ast.For):
                record_bad(stmt.target)
                walk_body(stmt.body, False)
                walk_body(stmt.orelse, False)
            elif isinstance(stmt, (ast.If, ast.While)):
                walk_body(stmt.body, False)
                walk_body(stmt.orelse, False)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                # context-manager targets bind names too; be conservative
                walk_body(stmt.body, False)
            elif isinstance(stmt, (ast.Try, )):
                walk_body(stmt.body, False)
                for h in stmt.handlers:
                    walk_body(h.body, False)
                walk_body(stmt.orelse, False)
                walk_body(stmt.finalbody, False)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Nested defs have their own scope; do not scan.
                continue

    walk_body(func_node.body, True)
    mutated = {n for n, c in counts.items() if c != 1}
    mutated.update(bad)
    return mutated


def find_escaped_names(func_node: ast.FunctionDef, call_arg_escapes=None) -> set:
    """Names passed as a *whole* argument to a call in a way that may
    mutate them or leak their address.

    The AST Store scan above only catches explicit Python rebinding.  The
    source language also passes typed locals *by output pointer/reference*
    to intrinsics that mutate them in place — e.g. the exported
    ``gmem_ld_v4s32(mem, index, r0, r1, r2, r3)`` emits
    ``gmem_ld_v4s32_fn(..., &r0, &r1, &r2, &r3)`` and stores through those
    pointers (``*r0 = value.x; ...``).  Such a local is initialized once in
    the AST (``r0: ll.int32 = 0``) yet written at runtime, so it must never
    be emitted ``static constexpr``: nvcc would reject taking ``&r0``.

    Whether a given call argument escapes is decided by ``call_arg_escapes``
    ``(call_node, arg_expr) -> bool``.  Code generation supplies a resolver
    that inspects the resolved builtin's codegen template and reports escape
    exactly when that argument position is emitted by address (``&{param}``)
    or cannot be proven read-by-value.  With no predicate, the conservative
    default treats *every* whole-``Name`` call argument as escaping; this is
    a safe over-approximation (it would also disqualify an extent passed to
    ``alloc_shared_memory``), so callers that can resolve builtins should
    always pass the precise predicate.  Keyword whole-Name arguments are
    handled identically.
    """
    escaped: set = set()

    def _default_escapes(call_node, arg_expr):
        return isinstance(arg_expr, ast.Name)

    escapes_arg = call_arg_escapes if call_arg_escapes is not None \
        else _default_escapes

    class _Escapes(ast.NodeVisitor):

        def visit_Call(self, node: ast.Call) -> None:
            for arg in node.args:
                if isinstance(arg, ast.Name) and escapes_arg(node, arg):
                    escaped.add(arg.id)
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name) and escapes_arg(node, kw.value):
                    escaped.add(kw.value.id)
            # Keep walking: nested calls / indices may contain further loads.
            self.generic_visit(node)

    _Escapes().visit(func_node)
    return escaped


class StaticIntConst:
    """Record for one provably-static integer local."""

    __slots__ = ("name", "value", "ll_type", "def_index")

    def __init__(self, name: str, value: int, ll_type: LLType, def_index: int):
        self.name = name
        self.value = value
        self.ll_type = ll_type
        # Index of the defining top-level statement; a use inside a
        # statement that precedes the definition may not rely on it.
        self.def_index = def_index


def _resolve_annotation_type(ann_node: ast.AST, resolve):
    """Best-effort LLType for an annotation node; None when unresolvable."""
    try:
        t = resolve(ann_node)
    except Exception:
        return None
    return t if isinstance(t, LLType) else None


def compute_static_int_consts(func_node: ast.FunctionDef, resolve_annotation, infer_expr_type,
                              extra_disqualified=None) -> Dict[str, StaticIntConst]:
    """Compute provably-static integer locals of one function.

    Parameters
    ----------
    func_node : ast.FunctionDef
        The function whose body is analyzed (top-level statements only are
        candidate definitions).
    resolve_annotation : callable(ast.AST) -> LLType
        Resolves an annotation node to an LLType (e.g. the emitter's
        TypeInferencer annotation resolver).
    infer_expr_type : callable(ast.AST, Dict[str, LLType]) -> Optional[LLType]
        Infers an expression node's LLType given the current scope map.

    Returns
    -------
    Dict[str, StaticIntConst]
        Mapping name -> record, in definition order.  Only names that
        pass every constness check are present.
    """
    params = {arg.arg for arg in func_node.args.args}
    mutated = find_mutated_names(func_node)
    # Names whose value escapes through a call argument that may take its
    # address or mutate it (see find_escaped_names / the codegen-supplied
    # escape set) are never provably-static, even with a single AST Store.
    disqualified = set(mutated)
    if extra_disqualified:
        disqualified.update(extra_disqualified)

    result: Dict[str, StaticIntConst] = {}
    bindings: Dict[str, int] = {}
    scope_lltypes: Dict[str, LLType] = {}

    for index, stmt in enumerate(func_node.body):
        # Candidate definition shapes; anything else leaves the maps
        # unchanged (uses are validated against def_index at emission).
        value_node = None
        ann_type: Optional[LLType] = None
        target_name: Optional[str] = None

        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target_name = stmt.target.id
            value_node = stmt.value
            ann_type = _resolve_annotation_type(stmt.annotation, resolve_annotation)
        elif (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            target_name = stmt.targets[0].id
            value_node = stmt.value
            ann_type = None

        if target_name is None:
            continue
        if (target_name in params or target_name in disqualified or target_name in result
                or target_name.startswith(_BANNED_NAME_PREFIXES)):
            continue

        # Constant-fold the initializer against earlier statics only.
        value = try_eval_const_int(value_node, bindings)
        if value is None:
            continue

        # Determine the integer type: explicit annotation wins; otherwise
        # ask the type inferencer with the statics seen so far.
        ll_type = ann_type
        if ll_type is None:
            ll_type = infer_expr_type(value_node, scope_lltypes)
        if not isinstance(ll_type, IntType):
            continue
        rng = _int_range(ll_type)
        if rng is None or not (rng[0] <= value <= rng[1]):
            continue

        rec = StaticIntConst(target_name, value, ll_type, index)
        result[target_name] = rec
        bindings[target_name] = value
        scope_lltypes[target_name] = ll_type

    return result
