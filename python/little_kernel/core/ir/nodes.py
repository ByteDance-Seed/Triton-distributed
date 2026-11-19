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
"""Typed lowered AST/IR node definitions (RF06).

The nodes subclass the standard ``ast.expr``/``ast.stmt`` bases so they plug
into the existing AST pipeline unchanged: ``ast.walk``, ``copy.deepcopy``,
``ast.copy_location``, ``ast.dump`` and the codegen/inference visitors all
work on them. Unlike the legacy ``_lk_cpp_*`` attributes, every node is an
explicit type with declared result type, effect and source span, and every
embedded backend spelling is validated at construction time.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable, Optional

from little_kernel.core.type_system import LLType, Pointer

from little_kernel.core.ir.spans import SourceSpan, span_from_node
from little_kernel.core.ir.effects import IROpEffect, coerce_effect

#: Where a :class:`TargetStatement` is emitted.
TARGET_SECTIONS = ("main", "builtin")

#: Address spaces a :class:`ResourceReference` may name. These mirror the
#: backend memory spaces (see ``core.memory_effects.MemorySpace``) plus an
#: explicit generic-address case.
RESOURCE_SPACES = ("gmem", "smem", "tmem", "rmem", "cmem", "generic")

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
# A qualified C++ lvalue path such as ``__lk_smem.buffer`` or ``foo.bar.baz``,
# optionally with a single trailing ``[index]`` where the index is a validated
# identifier or non-negative integer literal (never arbitrary text).
_DOTTED_IDENT = re.compile(rf"{_IDENT}(?:\.{_IDENT})*(?:\[(?:{_IDENT}|[0-9]+)\])?\Z")
_INCLUDE_RE = re.compile(r'(?:\A<[^<>"\r\n]+>\Z)|(?:\A"[^<>\r\n"]+"\Z)')
_IDENT_RE = re.compile(rf"\A{_IDENT}\Z")


def _require_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _require_lltype(value: Any, what: str) -> LLType:
    if not isinstance(value, LLType):
        raise TypeError(f"{what} must be an LLType, got {type(value).__name__}: {value!r}")
    return value


def _require_ident_path(value: str, what: str) -> str:
    value = _require_text(value, what)
    if not is_valid_ident_path(value):
        raise ValueError(f"{what} must be a C++ identifier/dotted path with an optional "
                         f"single identifier-or-literal index, got {value!r}")
    return value


def is_valid_ident_path(value: Any) -> bool:
    """True when ``value`` is a validated C++ lvalue path.

    An identifier or dotted member path (``a``, ``a.b.c``), optionally with a
    single trailing ``[index]`` whose index is an identifier or non-negative
    integer literal. Arbitrary text (operators, calls, commas) is rejected so
    unvalidated strings can never be emitted as ordinary names.
    """
    return isinstance(value, str) and bool(_DOTTED_IDENT.match(value))


def _bind_location(node: ast.AST, span: SourceSpan) -> None:
    """Mirror a span onto the standard ast location attributes.

    This keeps every existing consumer that reads ``node.lineno`` (error
    reporting, ``ast.copy_location``, dumps with attributes) working without
    teaching it about :class:`SourceSpan`.
    """
    node.span = span
    node.lineno = span.lineno
    node.col_offset = 0 if span.col_offset is None else span.col_offset
    node.end_lineno = span.end_lineno
    node.end_col_offset = (0 if span.end_col_offset is None else span.end_col_offset)


class LoweredIRNode(ast.AST):
    """Base class for typed lowered operations."""

    is_lowered_ir = True
    # Surface the standard ast location attributes in ast.dump / traversal.
    _attributes = ("lineno", "col_offset", "end_lineno", "end_col_offset")

    @property
    def source_span(self) -> SourceSpan:
        return self.span

    def __deepcopy__(self, memo):
        """Clone a lowered node the same way the AST pipeline clones nodes.

        AST child fields are deep-copied (so transforms stay independent),
        while interned immutable LLType/effect/span values are shared rather
        than reconstructed.
        """
        return clone_lowered_node(self, _memo=memo)


class TargetExpression(LoweredIRNode, ast.expr):
    """An explicitly typed backend expression escape hatch.

    ``spelling`` is emitted verbatim at the backend boundary but is validated
    as non-empty text at construction; ``result_type`` is the declared
    :class:`LLType` of the value (including pointer/address-space-bearing
    types), never an unvalidated ``int32`` default.
    """

    _fields = ("spelling", "result_type", "effect", "span", "symbol")

    def __init__(self, spelling: Optional[str] = None, result_type: Optional[LLType] = None,
                 effect: IROpEffect = IROpEffect.ESCAPE, span: Optional[SourceSpan] = None,
                 symbol: Optional[str] = None, *, source: Optional[ast.AST] = None):
        self.spelling = (_require_text(spelling, "TargetExpression spelling") if spelling is not None else None)
        self.result_type = (_require_lltype(result_type, "result_type") if result_type is not None else None)
        self.effect = coerce_effect(effect)
        if symbol is not None:
            _require_ident_path(symbol, "symbol")
        self.symbol = symbol
        _bind_location(self, span if span is not None else span_from_node(source))


class TargetStatement(LoweredIRNode, ast.stmt):
    """A backend statement emitted into the kernel body (or builtin section).

    Statements carry a declared effect; a statement with no effect (pure) is
    malformed because it would be dead code the verifier could not reason
    about.
    """

    _fields = ("spelling", "effect", "section", "span", "symbol")

    def __init__(self, spelling: Optional[str] = None, effect: IROpEffect = IROpEffect.ESCAPE, section: str = "main",
                 span: Optional[SourceSpan] = None, symbol: Optional[str] = None, *, source: Optional[ast.AST] = None):
        self.spelling = (_require_text(spelling, "TargetStatement spelling") if spelling is not None else None)
        self.effect = coerce_effect(effect)
        if self.effect is IROpEffect.PURE:
            raise ValueError("TargetStatement must declare a side effect (not pure)")
        if section not in TARGET_SECTIONS:
            raise ValueError(f"section must be one of {TARGET_SECTIONS}, got {section!r}")
        self.section = section
        if symbol is not None:
            _require_ident_path(symbol, "symbol")
        self.symbol = symbol
        _bind_location(self, span if span is not None else span_from_node(source))


class TargetInclude(LoweredIRNode, ast.stmt):
    """A backend ``#include`` directive.

    ``include`` is the exact spelling following ``#include`` and must be an
    angle-bracket (``<foo/bar.cuh>``) or quoted (``"foo.h"``) include.
    """

    _fields = ("include", "span")

    def __init__(self, include: Optional[str] = None, span: Optional[SourceSpan] = None, *,
                 source: Optional[ast.AST] = None):
        if include is not None:
            include = _require_text(include, "include").strip()
            if not _INCLUDE_RE.match(include):
                raise ValueError("include must be an angle-bracket <...> or quoted "
                                 f"\"...\" directive, got {include!r}")
        self.include = include
        _bind_location(self, span if span is not None else span_from_node(source))


class TargetHelper(LoweredIRNode, ast.stmt):
    """A backend helper / device-function definition.

    Emitted to the builtin/device-function section and deduplicated by
    ``name`` when given.
    """

    _fields = ("spelling", "name", "span")

    def __init__(self, spelling: Optional[str] = None, name: Optional[str] = None, span: Optional[SourceSpan] = None, *,
                 source: Optional[ast.AST] = None):
        self.spelling = (_require_text(spelling, "TargetHelper spelling") if spelling is not None else None)
        if name is not None:
            name = _require_text(name, "name").strip()
            if not _IDENT_RE.match(name):
                raise ValueError(f"helper name must be a C++ identifier, got {name!r}")
        self.name = name
        _bind_location(self, span if span is not None else span_from_node(source))


class ResourceReference(LoweredIRNode, ast.expr):
    """A typed reference to a lowered memory/register resource.

    The result type is ``Pointer[element_type]`` (a 64-bit address-bearing
    pointer with the declared address ``space``), so pointer results are never
    collapsed to ``int32``. ``resource`` is a validated C++ lvalue path
    (identifier or dotted member path); it is never an arbitrary raw string.
    """

    _fields = ("resource", "element_type", "space", "result_type", "span", "symbol")

    def __init__(self, resource: Optional[str] = None, element_type: Optional[LLType] = None, space: str = "generic",
                 result_type: Optional[LLType] = None, span: Optional[SourceSpan] = None, symbol: Optional[str] = None,
                 *, source: Optional[ast.AST] = None):
        if resource is not None:
            self.resource = _require_ident_path(resource, "resource")
        else:
            self.resource = None
        if space not in RESOURCE_SPACES:
            raise ValueError(f"space must be one of {RESOURCE_SPACES}, got {space!r}")
        self.space = space
        self.element_type = (_require_lltype(element_type, "element_type") if element_type is not None else None)
        # result_type is positional only to satisfy AST cloning (it is a
        # declared field); it is always (re)derived here so it can never be
        # spoofed into a non-pointer catch-all.
        self.result_type = (Pointer(self.element_type) if self.element_type is not None else None)
        if symbol is not None:
            _require_ident_path(symbol, "symbol")
        self.symbol = symbol if symbol is not None else self.resource
        self.effect = IROpEffect.READ
        _bind_location(self, span if span is not None else span_from_node(source))


class StageReference(LoweredIRNode, ast.expr):
    """A pipeline-stage-indexed view over a resource.

    Lowers to ``(base + (stage) * stride)`` where ``base`` is a
    :class:`ResourceReference` (or any lowered expression of pointer type) and
    ``stage`` is a lowered integer expression. The result retains the base's
    pointer/element type.
    """

    _fields = ("base", "stage", "stride", "element_type", "result_type", "span", "symbol")

    def __init__(self, base: Optional[ast.expr] = None, stage: Optional[ast.expr] = None, stride: int = 0,
                 element_type: Optional[LLType] = None, result_type: Optional[LLType] = None,
                 span: Optional[SourceSpan] = None, symbol: Optional[str] = None, *, source: Optional[ast.AST] = None):
        if base is not None and not isinstance(base, ast.expr):
            raise TypeError("StageReference base must be an AST expression")
        if stage is not None and not isinstance(stage, ast.expr):
            raise TypeError("StageReference stage must be an AST expression")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 0:
            raise ValueError(f"stride must be a non-negative integer, got {stride!r}")
        self.base = base
        self.stage = stage
        self.stride = stride

        inferred = getattr(base, "result_type", None)
        if element_type is None:
            if isinstance(inferred, Pointer):
                element_type = inferred.inner_type
            elif base is not None:
                raise TypeError("element_type is required when base does not declare a "
                                "pointer result type")
        self.element_type = (_require_lltype(element_type, "element_type") if element_type is not None else None)
        if self.element_type is not None and inferred is not None \
                and isinstance(inferred, Pointer) \
                and inferred.inner_type != self.element_type:
            raise TypeError(f"StageReference element_type {self.element_type} does not "
                            f"match base pointer element type {inferred.inner_type}")
        self.result_type = (Pointer(self.element_type) if self.element_type is not None else None)
        if symbol is not None:
            _require_ident_path(symbol, "symbol")
        self.symbol = symbol
        self.effect = IROpEffect.READ
        _bind_location(self, span if span is not None else span_from_node(source))


# ------------------------------------------------------------------------------
# Public factories (preferred over direct construction)
# ------------------------------------------------------------------------------


def target_expression(spelling, result_type, **kwargs) -> TargetExpression:
    return TargetExpression(spelling, result_type, **kwargs)


def target_statement(spelling, **kwargs) -> TargetStatement:
    return TargetStatement(spelling, **kwargs)


def target_include(include, **kwargs) -> TargetInclude:
    return TargetInclude(include, **kwargs)


def target_helper(spelling, **kwargs) -> TargetHelper:
    return TargetHelper(spelling, **kwargs)


def resource_reference(resource, element_type, **kwargs) -> ResourceReference:
    return ResourceReference(resource, element_type, **kwargs)


def stage_reference(base, stage, stride, **kwargs) -> StageReference:
    return StageReference(base, stage, stride, **kwargs)


# ------------------------------------------------------------------------------
# Introspection
# ------------------------------------------------------------------------------

_LOWERED_TYPES = (TargetExpression, TargetStatement, TargetInclude, TargetHelper, ResourceReference, StageReference)


def is_lowered_node(node: Any) -> bool:
    """Return True for any typed lowered IR node."""
    return isinstance(node, _LOWERED_TYPES)


def iter_lowered_nodes(tree: ast.AST) -> Iterable[LoweredIRNode]:
    """Yield every lowered IR node within ``tree`` (including ``tree``)."""
    if isinstance(tree, (list, tuple)):
        for item in tree:
            yield from iter_lowered_nodes(item)
        return
    for node in ast.walk(tree):
        if is_lowered_node(node):
            yield node


def clone_lowered_node(node: LoweredIRNode, *, children: Optional[dict] = None,
                       _memo: Optional[dict] = None) -> LoweredIRNode:
    """Clone a lowered node, sharing immutable identities.

    Immutable/interned values (LLType, :class:`SourceSpan`,
    :class:`IROpEffect`) are shared by identity; AST child fields and lists
    thereof are deep-copied unless a replacement is supplied via
    ``children`` (mapping field name -> new value). The source node is never
    mutated, so transforms that rebuild a child (e.g. constant folding) keep
    their input intact.
    """
    import copy as _copy
    if _memo is None:
        _memo = {}
    children = children if children is not None else {}
    new = type(node)()
    _memo[id(node)] = new
    for name in node._fields + node._attributes:
        if not hasattr(node, name):
            continue
        if name in children:
            new_value = children[name]
        else:
            value = getattr(node, name)
            if isinstance(value, (LLType, SourceSpan, IROpEffect)):
                new_value = value  # immutable/interned: share identity
            else:
                new_value = _copy.deepcopy(value, _memo)
        setattr(new, name, new_value)
    return new


def map_lowered_node_children(node: LoweredIRNode, visit) -> LoweredIRNode:
    """Rebuild ``node`` with ``visit`` applied to each AST child field.

    ``visit`` maps an ``ast.AST`` child to its replacement (returning the same
    node means "leave unchanged"). Primitive/immutable fields are shared.
    Returns a *new* node; the input is left unmodified. This is the single
    traversal contract used by AST mutators/passes so lowered nodes are never
    rewritten in place.
    """
    replacements = {}
    for name in node._fields:
        if not hasattr(node, name):
            continue
        value = getattr(node, name)
        if isinstance(value, ast.AST):
            replacements[name] = visit(value)
    return clone_lowered_node(node, children=replacements)


def grouped_target_spelling(spelling: str) -> str:
    """Conservatively parenthesize an opaque backend expression.

    A backend escape-hatch expression is treated as an atomic value by the
    surrounding typed expression tree: its internal operator precedence is
    unknown, so it must always be grouped when embedded in a larger C++
    expression (e.g. ``(1 + 2) * 3`` rather than ``1 + 2 * 3``). Semantic
    correctness takes precedence over incidental source spelling.
    """
    return f"({spelling})"
