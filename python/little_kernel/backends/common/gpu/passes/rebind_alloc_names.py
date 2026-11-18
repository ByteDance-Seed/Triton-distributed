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
# publish, distribute, sublicense, and/or sell copies of the Software,
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
Rebinding/versioning pass for re-declared ll.empty/ll.zeros/ll.full buffers.

Problem (measured in frozen controller full2 data as the generic
`ir_memory_analysis_duplicate` validity failure): the memory analyzer keys
allocations by variable name, so any second `x = ll.empty(...)` on an
already-recorded name fails the whole program.  But re-declaring a Python
name is ordinary re-binding: under Python semantics the later binding
shadows the earlier buffer object; uses before the re-declaration bind the
old buffer, uses after bind the new one.  This pass implements exactly that
for the traced AST by *versioning* the second and later declarations
(`x`, `x__lkv1`, `x__lkv2`, ...) and rewriting every reference according
to a conservative flow-sensitive current-binding map.  After the pass each
allocation site owns a unique name, so `MemoryAnalyzer`'s name-keyed
bookkeeping stays valid and the generated CUDA declares one C array per
binding, as the author wrote.

Legality contract:

1. Names whose `ll.empty`/`ll.zeros`/`ll.full` call sites appear at most
   once are never touched; for programs without duplicates the output AST
   unparses byte-identical to the input (pure backward-compat path).
2. Only bare-`Name` targets participate.  Subscript-keyed declarations
   (`a[0] = ll.empty(...)`, produced by flatten_empty) keep their previous
   behavior (unique names are already synthesised by flatten_empty; true
   duplicates there still fail downstream with the old diagnostic).
3. A reference whose binding depends on control flow (a re-declaration
   visible inside an if/for/while/try/match branch versus outside it) is
   statically unresolvable.  Such a reference raises the old
   ValueError/`ir_memory_analysis_duplicate` contract (exit status
   unchanged versus the previous compiler: those programs never compiled
   before either).  An unconditional re-declaration after the branch
   clears the ambiguous state, so e.g.

       x = ll.empty(...)
       if cond:
           x = ll.empty(...)
       x = ll.empty(...)   # dominates the join; binding is unambiguous
       use(x)

   compiles with `use(x)` bound to the third buffer, while

       x = ll.empty(...)
       if cond:
           x = ll.empty(...)
       use(x)              # binding depends on `cond`: unresolvable

   still raises.
4. Nested function/class/lambda scopes are opaque: tracing binds names
   from the entry scope, and allocations there are rejected downstream
   already, so the pass neither renames nor descends into them.
5. Nothing about descriptors is assumed: repeated declarations may differ
   in shape/dtype/scope; each version is an independent buffer with its
   own bytes and offset in its scope.

The pass must run after `flatten_empty` (so comprehension-expanded
declarations are materialised) and before `insert_mem_alloc` (which owns
`MemoryAnalyzer`).
"""

import ast
from typing import Any, Callable, Dict, List, Set

from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.common.gpu.language.intrin.allocation import empty, zeros, full
from little_kernel.core.passes.utils.add_parent_reference import add_parent_references
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute

_ALLOC_API_NAMES = {empty.__name__, zeros.__name__, full.__name__}

# Statement forms whose bodies may execute zero times or conditionally, so
# a re-binding inside them does not dominate the statements after them.
_BRANCHY = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.Match)

# Statement forms that introduce a fresh variable scope; the traced entry
# program only has one scope, so these are left completely untouched.
_SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_VERSION_SUFFIX = "__lkv"


def _is_alloc_call(node: ast.AST, ctx: Dict[str, Any]) -> bool:
    """True if `node` is a call to ll.empty/ll.zeros/ll.full (same
    resolution strategy as MemoryAnalyzer)."""
    if not isinstance(node, ast.Call):
        return False
    func = recursive_resolve_attribute(node.func, ctx)
    return (isinstance(func, Callable) and hasattr(func, "__name__") and func.__name__ in _ALLOC_API_NAMES)


class _AllocSiteCounter(ast.NodeVisitor):
    """Count `name = ll.empty/zeros/full` sites per bare-Name target and
    collect all Name identifiers in one function body (excluding nested
    scoped defs, which tracing treats as separate scopes)."""

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx
        self.counts: Dict[str, int] = {}
        self.all_names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.all_names.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and _is_alloc_call(node.value, self.ctx)):
            name = node.targets[0].id
            self.counts[name] = self.counts.get(name, 0) + 1
        self.generic_visit(node)

    # Do not descend into nested scopes.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def process(self, func: ast.FunctionDef) -> None:
        # Visit statements of `func` directly: NodeVisitor.visit() on the
        # FunctionDef itself would short-circuit at visit_FunctionDef.
        for stmt in func.body:
            self.visit(stmt)


class _ExprRenamer(ast.NodeTransformer):
    """Rewrite Name references through the current binding map.

    Only names with an entry in `mapping` are ever rewritten; all other
    names are returned unchanged.  Does not descend into nested scopes.
    """

    def __init__(self, mapping: Dict[str, str], bad_uses: List[ast.AST], ambiguous: Set[str]):
        self.mapping = mapping
        self.bad_uses = bad_uses
        self.ambiguous = ambiguous

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.ambiguous and isinstance(node.ctx, ast.Load):
            self.bad_uses.append(node)
            return node
        new_id = self.mapping.get(node.id)
        if new_id is not None and new_id != node.id:
            return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


class _Rebinder:
    """Flow-sensitive versioning of duplicated allocation names in one
    function body.

    `mapping[name]` is the identifier the name is currently bound to; names
    absent from the map have no binding rewrite.  `ambiguous` holds tracked
    names whose current binding depends on branch outcomes; a Load of an
    ambiguous name is recorded in `bad_uses` (the pass raises afterwards,
    preserving the previous failure contract) and an unconditional
    re-declaration clears the flag.
    """

    def __init__(self, ctx: Dict[str, Any], tracked: Set[str], taken_names: Set[str]):
        self.ctx = ctx
        self.tracked = tracked
        self.taken_names = set(taken_names)
        self.mapping: Dict[str, str] = {}
        self.ambiguous: Set[str] = set()
        self.bad_uses: List[ast.AST] = []
        self.version_counters: Dict[str, int] = {}

    # -- helpers -----------------------------------------------------------

    def _fresh_version(self, name: str) -> str:
        idx = self.version_counters.get(name, 0) + 1
        while True:
            candidate = f"{name}{_VERSION_SUFFIX}{idx}"
            if candidate not in self.taken_names:
                self.taken_names.add(candidate)
                self.version_counters[name] = idx
                return candidate
            idx += 1

    def _rename_expr(self, node):
        if node is None:
            return None
        return _ExprRenamer(self.mapping, self.bad_uses, self.ambiguous).visit(node)

    # -- statement processing ----------------------------------------------

    def _merge(self, branch_maps: List[Dict[str, str]], entry_map: Dict[str, str], entry_amb: Set[str]) -> None:
        """Merge per-branch binding maps: bindings that differ across paths
        become ambiguous; bindings that agree on every path survive."""
        merged: Dict[str, str] = {}
        divergent: Set[str] = set()
        keys: Set[str] = set()
        for m in branch_maps:
            keys.update(m.keys())
        for k in keys:
            values = {m.get(k) for m in branch_maps}
            if len(values) == 1:
                merged[k] = values.pop()
            else:
                divergent.add(k)
        self.mapping = merged
        self.ambiguous = entry_amb | {k for k in divergent if k in self.tracked}

    def _process_branchy(self, stmt: ast.stmt) -> ast.stmt:
        entry_map = dict(self.mapping)
        entry_amb = set(self.ambiguous)
        branch_maps: List[Dict[str, str]] = []

        def run_suite(body: List[ast.stmt]) -> List[ast.stmt]:
            self.mapping = dict(entry_map)
            self.ambiguous = set(entry_amb)
            new_body = [self._process_stmt(s) for s in body]
            branch_maps.append(dict(self.mapping))
            return new_body

        if isinstance(stmt, ast.Match):
            stmt.subject = self._rename_expr(stmt.subject)
            for case in stmt.cases:
                if case.guard is not None:
                    self.mapping = dict(entry_map)
                    self.ambiguous = set(entry_amb)
                    case.guard = self._rename_expr(case.guard)
                case.body = run_suite(case.body)
            # No case may match at all (irrefutable catch-alls are not
            # distinguished here): the statement can be a no-op.
            branch_maps.append(dict(entry_map))
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            stmt.body = run_suite(stmt.body)
            if stmt.orelse:
                # orelse runs only when the try body completes normally.
                self.mapping = dict(branch_maps[-1])
                self.ambiguous = set(entry_amb)
                stmt.orelse = [self._process_stmt(s) for s in stmt.orelse]
                branch_maps[-1] = dict(self.mapping)
            for handler in stmt.handlers:
                if handler.type is not None:
                    self.mapping = dict(entry_map)
                    self.ambiguous = set(entry_amb)
                    handler.type = self._rename_expr(handler.type)
                handler.body = run_suite(handler.body)
            if stmt.finalbody:
                # finally executes on every outcome: merge bodies first,
                # then thread the finally suite through (its re-bindings
                # dominate everything after the statement).
                self._merge(branch_maps, entry_map, entry_amb)
                stmt.finalbody = [self._process_stmt(s) for s in stmt.finalbody]
                return stmt
        else:  # If / For / AsyncFor / While
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                stmt.iter = self._rename_expr(stmt.iter)
            elif isinstance(stmt, (ast.If, ast.While)):
                stmt.test = self._rename_expr(stmt.test)
            stmt.body = run_suite(stmt.body)
            if stmt.orelse:
                stmt.orelse = run_suite(stmt.orelse)
            else:
                # No else: the body may not execute at all, equivalent to
                # an empty orelse taken with the entry bindings.
                branch_maps.append(dict(entry_map))

        # Merge: bindings that differ across paths are ambiguous; bindings
        # that agree on every path survive.
        self._merge(branch_maps, entry_map, entry_amb)
        return stmt

    def _process_stmt(self, stmt: ast.stmt) -> ast.stmt:
        # Allocation (re-)declaration with a bare Name target.  The RHS is
        # evaluated before the binding, so it renames under binding state
        # that predates this statement.
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and _is_alloc_call(stmt.value, self.ctx)):
            name = stmt.targets[0].id
            if name not in self.tracked:
                # First/only declaration of a non-duplicated name: nothing
                # to do (pure backward-compat path); rename the RHS under
                # current bindings so shape expressions reference the right
                # versions of *other* tracked names.
                stmt.value = self._rename_expr(stmt.value)
                return stmt
            stmt.value = self._rename_expr(stmt.value)
            new_id = self._fresh_version(name)
            stmt.targets[0] = ast.copy_location(ast.Name(id=new_id, ctx=stmt.targets[0].ctx), stmt.targets[0])
            self.mapping[name] = new_id
            self.ambiguous.discard(name)
            return stmt

        if isinstance(stmt, _BRANCHY):
            return self._process_branchy(stmt)

        # With: the body always executes after __enter__; thread through.
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                item.context_expr = self._rename_expr(item.context_expr)
            stmt.body = [self._process_stmt(s) for s in stmt.body]
            return stmt

        # Ordinary assignment: renames RHS, then treats Name targets as
        # plain re-bindings (subsequent references bind the new value; its
        # identifier is unchanged because only tracked names are rewritten
        # and a plain value is not an allocation).
        if isinstance(stmt, ast.Assign):
            stmt.value = self._rename_expr(stmt.value)
            for idx, target in enumerate(stmt.targets):
                if isinstance(target, ast.Name) and target.id in self.tracked:
                    self.mapping[target.id] = target.id
                    self.ambiguous.discard(target.id)
                else:
                    stmt.targets[idx] = self._rename_expr(target)
            return stmt

        if isinstance(stmt, ast.AugAssign):
            stmt.value = self._rename_expr(stmt.value)
            stmt.target = self._rename_expr(stmt.target)
            return stmt

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                stmt.value = self._rename_expr(stmt.value)
            return stmt

        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in self.tracked:
                    # Deleting a versioned name breaks static resolution;
                    # keep the old failure contract if it is ever used.
                    self.ambiguous.add(target.id)
                else:
                    self._rename_expr(target)
            return stmt

        # Generic statement: rewrite contained expression Names.  All
        # statement forms with nested suites are handled above, so the
        # expression renamer cannot cross statement boundaries here.
        return _ExprRenamer(self.mapping, self.bad_uses, self.ambiguous).visit(stmt)

    def _process_body(self, body: List[ast.stmt]) -> List[ast.stmt]:
        processed = []
        for stmt in body:
            if isinstance(stmt, _SCOPED):
                processed.append(stmt)  # nested scope: untouched
            else:
                processed.append(self._process_stmt(stmt))
        return processed

    def error_message(self, func_name: str) -> str:
        uses = ", ".join(ast.unparse(u) for u in self.bad_uses[:3]) or "<unknown>"
        names = sorted(self.ambiguous)
        return (f"ll.empty call for {names[0] if names else '?'}"
                f" has already been recorded (ir_memory_analysis_duplicate);"
                f" the binding of {names} in function `{func_name}` depends"
                f" on branch or loop control flow (e.g. at `{uses}`) and"
                f" cannot be versioned statically. Restructure the frontend"
                f" so buffer re-declarations are unconditional, or use"
                f" distinct names for distinct buffers.")


def rebind_alloc_names(tree: ast.AST, ctx: Dict[str, Any]) -> ast.AST:
    """Version duplicated ll.empty/ll.zeros/ll.full names in the entry
    function so each allocation site is uniquely named (see module docs).
    Programs without duplicated declaration names are returned unchanged."""
    add_parent_references(tree)
    if __LITTLE_KERNEL_ENTRY__ not in ctx:
        # No kernel entry function in this compilation unit (e.g. struct
        # stubs): nothing to version.  Mirrors the guard in InsertMemAlloc.
        return tree
    entry_name = ctx[__LITTLE_KERNEL_ENTRY__].__name__
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_name:
            counter = _AllocSiteCounter(ctx)
            counter.process(node)
            tracked = {n for n, c in counter.counts.items() if c > 1}
            if not tracked:
                continue
            rebinder = _Rebinder(ctx, tracked, counter.all_names)
            node.body = rebinder._process_body(node.body)
            if rebinder.bad_uses:
                raise ValueError(rebinder.error_message(node.name))
    ast.fix_missing_locations(tree)
    return tree
