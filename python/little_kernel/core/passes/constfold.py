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

import ast
import copy as _copy
from enum import Enum
from typing import Dict, Any, Callable

from little_kernel.core.passes.pass_base import CompleteASTMutator
from little_kernel.core.passes.utils.preserve_attributes import create_node_with_attrs
from little_kernel.language.builtin_base import BUILTIN_ATTR, CONST_FUNC_ATTR
from little_kernel.core.type_system import LLType, ConstTypeHelper, ConstAnnotationType, const_annotation
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.error_report import raise_pass_error, get_source_from_ctx
from little_kernel.core.passes.utils.scope_manager import ScopeManager


class ConstantFolder(CompleteASTMutator):

    def __init__(self, ctx=None, all_types=None):
        super().__init__()
        self.ctx = ctx if ctx is not None else {}
        self.scope_manager = ScopeManager(self.ctx)
        self.defines = {}  # Global constants (from ctx) or folded local constants
        # self.kernel_defines was removed in favor of ScopeManager
        self.is_rvalue = False
        self.is_rvalue_stack = [False]
        self.in_kernel_function = False  # Track if we're inside a kernel function
        self.source_lines = None  # Cache source lines for error reporting
        # Map AST comparison operators to their corresponding Python operations
        self._op_mapping = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }
        self._op_handlers: Dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: lambda a, b: a + b,  # a + b
            ast.Sub: lambda a, b: a - b,  # a - b
            ast.Mult: lambda a, b: a * b,  # a * b
            ast.Div: lambda a, b: a / b,  # a / b
            ast.FloorDiv: lambda a, b: a // b,  # a // b
            ast.Mod: lambda a, b: a % b,  # a % b
            ast.Pow: lambda a, b: a**b,  # a **b
            ast.LShift: lambda a, b: a << b,  # a << b
            ast.RShift: lambda a, b: a >> b,  # a >> b
            ast.BitOr: lambda a, b: a | b,  # a | b
            ast.BitXor: lambda a, b: a ^ b,  # a ^ b
            ast.BitAnd: lambda a, b: a & b,  # a & b
            ast.MatMult: lambda a, b: a @ b,  # a @ b (matrix multiplication)
        }
        self._bool_handlers: Dict[type[ast.operator], Callable[[Any, Any], bool]] = {
            ast.And: lambda *values: all(values),
            ast.Or: lambda *values: any(values),
        }
        # Map AST unary operators to their corresponding Python operations.
        # Used to fold unary expressions whose operand is a compile-time
        # constant (e.g. "-5" -> Constant(-5), "~3" -> Constant(-4),
        # "not False" -> Constant(True)).
        self._unary_handlers: Dict[type[ast.unaryop], Callable[[Any], Any]] = {
            ast.UAdd: lambda a: +a,  # +a
            ast.USub: lambda a: -a,  # -a
            ast.Invert: lambda a: ~a,  # ~a
            ast.Not: lambda a: not a,  # not a
        }
        self.all_types = all_types if all_types is not None else {}

    def switch_on_use(self):
        self.is_rvalue_stack.append(self.is_rvalue)
        self.is_rvalue = True

    def switch_off_use(self):
        assert len(self.is_rvalue_stack) > 0
        self.is_rvalue = self.is_rvalue_stack.pop()

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """
        Fold Compare nodes into Constants if all operands are Constants.
        Example: 3 < 5 -> Constant(value=True)
        Example: 2 + 2 == 4 (after folding) -> Constant(value=True)
        """
        self.switch_on_use()
        # First, recursively process child nodes (in case they contain foldable expressions)
        processed_left = self.visit(node.left)
        processed_comparators = [self.visit(comp) for comp in node.comparators]
        processed_ops = node.ops  # Operators are not AST nodes to process
        self.switch_off_use()

        # Check if all operands are Constants after processing
        if not isinstance(processed_left, ast.Constant):
            # Left operand is not a constant → return original Compare (with processed children)
            new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                 comparators=processed_comparators)
            return new_compare

        for comp in processed_comparators:
            if not isinstance(comp, ast.Constant):
                # At least one comparator is not a constant → return original Compare
                new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                     comparators=processed_comparators)
                return new_compare

        # Extract constant values
        left_val = processed_left.value
        comparator_vals = [comp.value for comp in processed_comparators]

        # Evaluate chained comparisons (e.g., a < b < c → (a < b) and (b < c))
        try:
            result = True
            current_val = left_val
            for op, comp_val in zip(processed_ops, comparator_vals):
                # Get the operation function (e.g., Lt → lambda a,b: a < b)
                op_type = type(op)
                op_func = self._op_mapping.get(op_type)
                if not op_func:
                    # Unknown operator → cannot fold
                    raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

                # Evaluate current comparison step
                step_result = op_func(current_val, comp_val)
                if not step_result:
                    # Short-circuit: if any step fails, overall result is False
                    result = False
                    break

                # For chained comparisons, next step uses current comparator as left value
                current_val = comp_val

        except Exception as e:
            # Handle errors (e.g., incompatible types like 5 < "string")
            print(f"Warning: Could not fold comparison: {e}")
            new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                 comparators=processed_comparators)
            return new_compare

        # Return folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original location info for consistency
        ast.copy_location(folded_constant, node)

        return folded_constant

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """
        Fold BinOp nodes into Constants if both operands are Constants.
        Example: 2 + 3 → Constant(value=5)
        Example: (4 * 5) - 6 → Constant(value=14)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_left = self.visit(node.left)
        processed_right = self.visit(node.right)
        self.switch_off_use()

        # Check if both operands are Constants after processing
        if not (isinstance(processed_left, ast.Constant) and isinstance(processed_right, ast.Constant)):
            # At least one operand is not a constant → return original BinOp (with processed children)
            new_binop = create_node_with_attrs(ast.BinOp, node, left=processed_left, op=node.op, right=processed_right)
            return new_binop

        # Extract constant values
        left_val = processed_left.value
        right_val = processed_right.value
        op_type = type(node.op)

        # Evaluate the binary operation
        try:
            # Get the handler function for this operator (e.g., Add → lambda a,b: a+b)
            op_handler = self._op_handlers.get(op_type)
            if not op_handler:
                raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

            # Compute the result
            result = op_handler(left_val, right_val)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BinOp {ast.unparse(node)}: {e}")
            new_binop = create_node_with_attrs(ast.BinOp, node, left=processed_left, op=node.op, right=processed_right)
            return new_binop

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """
        Fold BoolOp nodes into Constants if both operands are Constants.
        Example: True and False → Constant(value=False)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_values = [self.visit(v) for v in node.values]
        self.switch_off_use()

        # Check if all operands are Constants after processing
        for processed_v in processed_values:
            if not isinstance(processed_v, ast.Constant):
                # At least one operand is not a constant → return original BoolOp (with processed children)
                new_boolop = create_node_with_attrs(ast.BoolOp, node, op=node.op, values=processed_values)
                return new_boolop

        # Extract constant values
        const_values = [v.value for v in processed_values]
        op_type = type(node.op)

        # Evaluate the handler operation
        try:
            # Get the handler function for this operator
            op_handler = self._bool_handlers.get(op_type)
            if not op_handler:
                raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

            # Compute the result
            result = op_handler(*const_values)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BoolOp {ast.unparse(node)}: {e}")
            new_boolop = create_node_with_attrs(ast.BoolOp, node, op=node.op, values=processed_values)
            return new_boolop

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    @staticmethod
    def _stmt_bound_names(stmt: ast.AST):
        """Names bound by ``stmt`` through any binding form.

        Catches every way Python binds a local: ``Name`` targets with a
        ``Store``/``Del`` context (plain ``x = ...``, tuple/list target
        elements, walrus ``x := ...``, for/with targets, plain-Name
        ``x += 1``), ``import x`` / ``from m import x`` aliases (bound via
        ``ast.alias``, never a ``Name(Store)``), nested ``def``/``class``
        statements (which bind their own name without any ``Name(Store)``
        node), ``except ... as x`` names, ``match`` capture names, and
        ``global``/``nonlocal`` declarations, conservatively treated as
        rebinding the local.  Used both to disqualify a workspace base whose
        name is reassigned (the base then names a different buffer) and to
        invalidate folds whose stored expression reads a name rebound between
        store and load.
        """
        out = set()

        class _B(ast.NodeVisitor):

            def visit_Name(self, n):
                if isinstance(n.ctx, (ast.Store, ast.Del)):
                    out.add(n.id)

            def visit_FunctionDef(self, n):
                # ``def kvc(): ...`` rebinds the local ``kvc`` to the
                # function object; its name is not a Name(Store) node.
                out.add(n.name)
                self.generic_visit(n)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, n):
                out.add(n.name)
                self.generic_visit(n)

            def visit_Import(self, n):
                for alias in n.names:
                    # ``import a.b`` binds ``a``; ``import a.b as c`` binds
                    # ``c``.
                    out.add(alias.asname or alias.name.split(".")[0])

            def visit_ImportFrom(self, n):
                for alias in n.names:
                    if alias.name == "*":
                        # Star imports bind an unknown set of locals; the
                        # caller cannot isolate them, so treat nothing here
                        # and let the base-disqualification stay conservative
                        # only for names it can see.  A star import in a
                        # function is a SyntaxError in module-level compile,
                        # but this pass runs source-to-source.
                        continue
                    out.add(alias.asname or alias.name)

            def visit_ExceptHandler(self, n):
                if n.name:
                    out.add(n.name)
                self.generic_visit(n)

            # ``match`` capture patterns bind their target names without a
            # Name(Store) node (``case x:`` binds ``x``, ``case [*rest]:``
            # binds ``rest``, ``case {**rest}:`` binds ``rest``).
            def visit_MatchAs(self, n):
                if n.name:
                    out.add(n.name)
                self.generic_visit(n)

            def visit_MatchStar(self, n):
                if n.name:
                    out.add(n.name)
                self.generic_visit(n)

            def visit_MatchMapping(self, n):
                if n.rest:
                    out.add(n.rest)
                self.generic_visit(n)

            def visit_Global(self, n):
                out.update(n.names)

            def visit_Nonlocal(self, n):
                out.update(n.names)

        _B().visit(stmt)
        return out

    # ------------------- workspace-cell memoization -------------------
    # Cross-tree scalar memoization rule: a load from a GMEM workspace cell
    # whose only definition in the enclosing function is a single store of a
    # scalar expression resolves to that stored expression.  This is the
    # const-fold/tensor-value-arm of the single-definition store->load chain:
    # when a frontend smuggles a scalar through a GMEM cell (``kvc[0] = kv0``)
    # and a later expression tree reads the same cell (``_kv = kvc[b]`` with
    # ``b`` a constant), folding the load to the stored scalar keeps the
    # documented value attached to downstream uses (e.g. a dynamic loop trip
    # count), instead of collapsing to a conservative fallback.

    @staticmethod
    def _subscript_const_index(node: ast.Subscript, const_names=None):
        """Constant index of a subscript, or None when not provably constant.

        ``const_names`` maps local names to integer literals they are bound
        to at this program point (plain ``b = 1`` or ``b: const = 1``): a
        ``Name`` slice resolves through it.  Everything else — expressions,
        slices, bools — is not a provable cell index here.
        """
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, int) \
                and not isinstance(sl.value, bool):
            return sl.value
        if const_names and isinstance(sl, ast.Name):
            return const_names.get(sl.id)
        return None

    @staticmethod
    def _is_pure_scalar_expr(expr: ast.AST) -> bool:
        """True when ``expr`` is duplicable: names/constants and arithmetic.

        The workspace-cell fold splices a deep copy of the stored expression
        into every dominated load while *keeping* the defining store, so any
        node with a possible side effect or its own memory read would be
        evaluated twice (a miscompile for impure callees such as RNG,
        atomics or runtime intrinsics, silently duplicated work otherwise).
        Only ``Constant``/``Name`` leaves over ``BinOp``/``UnaryOp``/
        ``BoolOp``/``Compare`` structure qualify; ``Call``, ``Subscript``,
        ``Attribute``, ``Yield``/``Await`` and everything else reject the
        cell.
        """
        allowed = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.operator, ast.unaryop,
                   ast.boolop, ast.cmpop, ast.Load, ast.expr_context)
        for child in ast.walk(expr):
            if not isinstance(child, allowed):
                return False
            # NamedTuple/etc. are fine; only plain Names read locals.
        return True

    @staticmethod
    def _collect_workload_cell_facts(func_node: ast.FunctionDef, const_names=None):
        """Pre-pass over a function body collecting GMEM workspace-cell facts.

        Returns ``(stores, dynamic_bases, called_bases)``:
        * ``stores``: ``base_name -> {const_index -> [definition, ...]}`` for
          every Store/Del-context subscript of a plain-Name base anywhere in
          the body, recorded uniformly in ``visit_Subscript`` so no enclosing
          statement or target form can hide a cell write: plain ``kvc[0] =
          v``, chained ``a = kvc[0] = v``, annotated ``kvc[0]: int = v``,
          sequence-unpacking targets (``kvc[0], a = ...``, ``[kvc[0],
          *rest] = ...``), loop/with targets (``for kvc[0] in it``, ``with r
          as kvc[0]``), ``del kvc[0]``/``del (kvc[0], a)``, read-modify-write
          ``kvc[0] += 1``, and stores nested in control-flow blocks.  The
          caller requires the recorded definition count per folded cell to
          equal the modeled top-level store count, so any store the narrower
          ``top_store`` collector missed counts against uniqueness and
          disables the fold.
        * ``dynamic_bases``: base names that are ever subscripted with a
          non-constant (or complex) index, subscripted in a store context
          we cannot model, rebound as a plain name — or read whole anywhere
          outside the tracked subscript-base position (``alias = kvc`` and
          any other untracked Name load), which creates an alias whose
          writes would be recorded under the alias's key, never under the
          base's — such bases alias conservatively.
        * ``called_bases``: base names passed whole to any call (a callee may
          mutate the buffer in place, so the single-definition proof is void).

        A base present in ``dynamic_bases`` or ``called_bases`` is entirely
        ineligible; the caller intersects before trusting ``stores``.
        ``const_names`` resolves single-definition literal-bound locals used
        as subscript indices (e.g. ``b: const = 1``); a ``Name`` slice absent
        from it is dynamic.
        """
        stores = {}
        dynamic_bases = set()
        called_bases = set()

        def _untracked_base_load(node):
            """Base of an eligible Name-load, or None when the name is tracked.

            A ``Name(ctx=Load)`` whose parent is a ``Subscript`` with the name
            as its direct base is the tracked cell-access form that ``stores``
            and the fold model: the subscript visitor accounts for it, so it
            is not an escape.  Every other whole-name load — ``alias = kvc``,
            a call argument handled by visit_Call, a comparison, an attribute
            base ``kvc.fill`` — hands (or may hand) the buffer out of the
            store->load chain, voiding the single-definition proof the same
            way a call argument does.  Function parameters never appear as
            Load-context names in this AST, so declarations are unaffected.
            """
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                return None
            parent = getattr(node, '_lk_parent', None)
            if (parent is not None and isinstance(parent, ast.Subscript) and parent.value is node):
                return None
            return node.id

        class _Facts(ast.NodeVisitor):

            def generic_visit(self, node):
                # Any binding of the workspace base name itself — plain
                # ``kvc = other``, tuple targets, walrus, plain-Name
                # ``kvc += delta``, for/with targets, imports, or
                # global/nonlocal declarations — repoints the name, so a
                # later ``kvc[0]`` no longer reads the buffer the store
                # wrote.  Treat such a base as aliased conservatively.
                if isinstance(node, ast.stmt) or isinstance(node, ast.expr):
                    bound = ConstantFolder._stmt_bound_names(node)
                    if bound:
                        dynamic_bases.update(bound)
                    # Aliasing escape: any whole-name read of a base outside
                    # the tracked subscript-base position — above all a
                    # binding RHS such as ``alias = kvc`` — creates a second
                    # name for the buffer.  Writes through the alias
                    # (``alias[0] = 9``) are recorded under the alias, never
                    # under ``kvc``, so the single-definition count for
                    # ``kvc`` would stay 1 and the fold would resurrect the
                    # stale stored value.  Treat the loaded base exactly like
                    # a call-escaped one.
                    escaped = _untracked_base_load(node)
                    if escaped is not None:
                        dynamic_bases.add(escaped)
                super().generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript):
                if isinstance(node.value, ast.Name):
                    base = node.value.id
                    if isinstance(node.ctx, (ast.Store, ast.Del)):
                        # Uniform store accounting: every subscript with a
                        # Store/Del context counts as a cell definition,
                        # regardless of the enclosing statement or target
                        # form — plain ``kvc[0] = v``, chained
                        # ``a = kvc[0] = v``, annotated ``kvc[0]: int = v``,
                        # sequence-unpacking targets (``kvc[0], a = ...``,
                        # ``[kvc[0], *rest] = ...``), loop/with targets
                        # (``for kvc[0] in it``, ``with r as kvc[0]``),
                        # ``del kvc[0]``/``del (kvc[0], a)``, and
                        # read-modify-write ``kvc[0] += 1`` (whose target is
                        # a Store-context subscript).  The parser sets the
                        # Store/Del context on the Subscript node itself in
                        # every one of these forms, so recording here means
                        # no enclosing target structure can hide a cell
                        # write from the single-definition proof.
                        self._record_cell_store(node, None)
                    else:
                        idx = ConstantFolder._subscript_const_index(node, const_names)
                        if idx is None:
                            # Non-constant slice: cannot prove which cell;
                            # the whole base becomes aliased conservatively.
                            dynamic_bases.add(base)
                self.generic_visit(node)

            def _record_cell_store(self, tgt: ast.Subscript, value):
                # Every subscript store of the base counts as a cell
                # definition.  Called from visit_Subscript for Store/Del-
                # context subscripts only, so each definition is recorded
                # exactly once no matter which statement or target form
                # carries it; the single-definition proof requires that the
                # total recorded count match the modeled top-level count.
                idx = ConstantFolder._subscript_const_index(tgt, const_names)
                if idx is None:
                    dynamic_bases.add(tgt.value.id)
                else:
                    stores.setdefault(tgt.value.id, {}).setdefault(idx, []).append(value)

            def visit_Call(self, node: ast.Call):
                # A method call on the base — ``kvc.fill(0)`` — hands the
                # buffer to the callee as its implicit first argument
                # (``Call.func`` is ``Attribute(value=Name(kvc), ...)`` and
                # never appears in ``args``), so it may mutate the buffer in
                # place exactly like a whole-name argument escape does.
                if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
                    called_bases.add(node.func.value.id)
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        called_bases.add(arg.id)
                    elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
                        # ``may_mutate(*kvc)`` splices the buffer's elements
                        # into the callee's positional arguments; that hands
                        # the buffer to arbitrary code exactly like the plain
                        # whole-name form ``may_mutate(kvc)`` does, so the
                        # single-definition proof is void.  The parser wraps a
                        # starred pass-through as Starred(value=Name(...)),
                        # never a bare Name in args.
                        called_bases.add(arg.value.id)
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Name):
                        called_bases.add(kw.value.id)
                self.generic_visit(node)

        # Parent pointers let the generic-visit escape scan distinguish a
        # tracked subscript-base load (``kvc`` in ``kvc[0]``, accounted for by
        # visit_Subscript) from an untracked whole-name read (``kvc`` in
        # ``alias = kvc``).  The attribute name is private to this pass.
        for parent in ast.walk(func_node):
            for child in ast.iter_child_nodes(parent):
                child._lk_parent = parent

        _Facts().visit(func_node)
        return stores, dynamic_bases, called_bases

    @staticmethod
    def _top_level_const_names(func_node: ast.FunctionDef):
        """Top-level locals bound to a plain int literal exactly once.

        Maps ``name -> int`` for statements ``b = 1`` / ``b: const = 1`` /
        ``b: <anything> = 1`` at top level where ``name`` is never rebound
        anywhere in the body.  Subscript indices written as such a name are
        provably the same cell as the literal; generic constant folding would
        rewrite the name to the literal anyway, so resolving through this map
        only anticipates that fold for cell matching.  Ambiguous (rebound)
        names are excluded entirely, keeping index resolution conservative.
        """
        bound = {}

        def _literal_int(assign_value):
            if (isinstance(assign_value, ast.Constant) and isinstance(assign_value.value, int)
                    and not isinstance(assign_value.value, bool)):
                return assign_value.value
            return None

        for stmt in func_node.body:
            name = None
            value = None
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                name, value = stmt.targets[0].id, stmt.value
            elif (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value):
                name, value = stmt.target.id, stmt.value
            if name is not None:
                bound.setdefault(name, []).append(_literal_int(value))

        rebound = {}
        for stmt in func_node.body:
            for name in ConstantFolder._stmt_bound_names(stmt):
                rebound[name] = rebound.get(name, 0) + 1

        return {
            name: vals[0]
            for name, vals in bound.items()
            if len(vals) == 1 and vals[0] is not None and rebound.get(name, 0) == 1
        }

    def _fold_workspace_cell_loads(self, func_node: ast.FunctionDef):
        """Rewrite eligible workspace-cell loads to their stored scalar value.

        Only top-level statements of ``func_node.body`` participate: a store
        ``<base>[<const>] = <expr>`` at top-level statement position ``i``
        documents every load ``<base>[<const>]`` in top-level statements at
        positions ``j > i`` (the store strictly dominates them).  Loads nested
        in control-flow blocks are left untouched — their execution is not
        provably dominated by the store under this cheap positional order —
        and bases with any dynamic-subscript or whole-name call argument are
        excluded by the fact pass.  Symmetrically, *stores* nested in
        control-flow blocks — or written in a form the positional model does
        not handle (chained ``a = kvc[0] = v``, annotated ``kvc[0]: int =
        v``, sequence-unpacking targets such as ``kvc[0], a = ...``,
        loop/with targets, ``del kvc[0]``, ``kvc[0] += 1``) — disqualify
        their cell: the fold requires the cell's total recorded definition
        count (:meth:`_collect_workload_cell_facts`, which sees every
        Store/Del-context subscript) to equal its modeled top-level store
        count, so exactly one plain top-level store may define the cell.
        Subscript indices may be plain int
        literals or top-level locals bound once to an int literal (``b: const
        = 1``; resolved via :meth:`_top_level_const_names`).  The stored
        expression must be duplicable — names/constants and arithmetic only
        (:meth:`_is_pure_scalar_expr`) — because the defining store is kept
        while a copy is spliced into each load.  Its free names must not be
        rebound between store and load; since names rebound in the interval
        could change its meaning, any top-level (re)binding of a free name in
        ``(i, j]`` disqualifies folds for that cell from that point on —
        including bindings inside statement ``j`` itself (a walrus target in
        its own RHS, say), because Python evaluates the statement
        left-to-right and the binding takes effect before the load executes.

        Returns a new statement list (the input is not mutated in place).
        """
        # Cheap guard: without any subscript there is no workspace cell.
        if not any(isinstance(n, ast.Subscript) for n in ast.walk(func_node)):
            return func_node.body

        const_names = self._top_level_const_names(func_node)
        stores, dynamic_bases, called_bases = \
            self._collect_workload_cell_facts(func_node, const_names)
        # base -> {idx -> value_node}: single stores at top level only.
        top_store = {}
        for idx, stmt in enumerate(func_node.body):
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript)
                    and isinstance(stmt.targets[0].value, ast.Name) and isinstance(stmt.targets[0].ctx, ast.Store)):
                tgt = stmt.targets[0]
                cell_idx = self._subscript_const_index(tgt, const_names)
                if (cell_idx is not None and self._is_pure_scalar_expr(stmt.value)):
                    top_store.setdefault(tgt.value.id, {})
                    top_store[tgt.value.id].setdefault(cell_idx, []).append((idx, stmt.value))

        if not top_store:
            return func_node.body

        # Single-definition uniqueness must hold across *every* store form
        # and nesting depth, not only the modeled top-level plain Assign.
        # ``top_store`` covers ``<base>[<const>] = <expr>`` at top-level
        # statement position; ``stores`` from the fact pass records every
        # Store/Del-context subscript — chained targets (``a = kvc[0] =
        # v``), annotated stores (``kvc[0]: int = v``), sequence-unpacking
        # targets (``kvc[0], a = ...``, ``[kvc[0], *rest] = ...``), loop/with
        # targets (``for kvc[0] in it``, ``with r as kvc[0]``), ``del
        # kvc[0]``/``del (kvc[0], a)``, read-modify-write ``kvc[0] += 1``,
        # and stores nested inside ``for``/``while``/``if``/``try`` blocks.
        # Any of those makes the cell's total definition count differ from
        # the top-level count, so requiring the counts to match disqualifies
        # such cells wholesale — the symmetric conservative rule to this
        # pass refusing to fold loads nested in control flow.
        for base, cells in list(top_store.items()):
            for cell_idx, candidates in list(cells.items()):
                total = len(stores.get(base, {}).get(cell_idx, []))
                if total != len(candidates):
                    del top_store[base][cell_idx]
            if not top_store[base]:
                del top_store[base]
        if not top_store:
            return func_node.body

        def _free_names(expr):
            names = set()

            class _Names(ast.NodeVisitor):

                def visit_Name(self, n):
                    names.add(n.id)

            _Names().visit(expr)
            return names

        class _LoadFold(ast.NodeTransformer):

            def __init__(self, cell_values):
                self.cell_values = cell_values  # base -> {idx -> value_node}

            def visit_Subscript(self, node):
                self.generic_visit(node)
                if (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)):
                    base = node.value.id
                    cell_idx = ConstantFolder._subscript_const_index(node, const_names)
                    if (base in self.cell_values and cell_idx in self.cell_values[base]):
                        # Deep-copy: one stored expression may document many
                        # load sites, and downstream fold mutations must not
                        # alias across them.
                        value = _copy.deepcopy(self.cell_values[base][cell_idx])
                        return ast.copy_location(value, node)
                return node

            # Do not descend into nested control flow: positional dominance
            # proof only covers top-level statements.
            def _keep(self, node):
                return node

            visit_If = _keep
            visit_For = _keep
            visit_While = _keep
            visit_With = _keep
            visit_Try = _keep
            visit_FunctionDef = _keep
            visit_AsyncFunctionDef = _keep
            visit_ClassDef = _keep

        new_body = []
        # Active folds: base -> {cell_idx -> value_node}
        active = {}
        for pos, stmt in enumerate(func_node.body):
            fold_map = {}
            for base, cells in top_store.items():
                if base in dynamic_bases or base in called_bases:
                    continue
                for cell_idx, candidates in cells.items():
                    if len(candidates) != 1:
                        continue
                    store_pos, value_node = candidates[0]
                    if store_pos < pos:
                        fold_map.setdefault(base, {})[cell_idx] = value_node
            if fold_map:
                active = fold_map
            # Names bound anywhere in this statement — including walrus
            # targets inside its own RHS — must disqualify folds of the
            # statement itself, not only later ones: Python evaluates a
            # statement left-to-right, so ``n = (x := 5) + kvc[0]`` rebinds
            # ``x`` *before* the cell load executes, while the store the
            # fold substitutes from saw the old ``x``.  Poison those cells
            # (the statement binds the name regardless of branch outcomes at
            # top level) and fold only with the surviving map.
            rebound = self._stmt_bound_names(stmt)
            if rebound and active:
                for base in list(active):
                    for cell_idx in list(active[base]):
                        value_node = active[base][cell_idx]
                        if _free_names(value_node) & rebound:
                            active[base].pop(cell_idx)
                            if base in top_store and cell_idx in top_store[base]:
                                top_store[base][cell_idx] = []  # poison
                    if not active[base]:
                        active.pop(base)
            if active:
                stmt = _LoadFold(active).visit(stmt)
            new_body.append(stmt)

        return new_body

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        """
        Fold UnaryOp nodes into Constants if the operand is a Constant.
        Example: -5 → Constant(value=-5)
        Example: ~3 → Constant(value=-4)
        Example: not False → Constant(value=True)
        """
        # First, recursively process the operand (handles nested expressions)
        self.switch_on_use()
        processed_operand = self.visit(node.operand)
        self.switch_off_use()

        # If the operand is not a constant, we cannot fold → return a UnaryOp
        # with the processed operand.
        if not isinstance(processed_operand, ast.Constant):
            new_unaryop = create_node_with_attrs(ast.UnaryOp, node, op=node.op, operand=processed_operand)
            return new_unaryop

        # Extract the constant operand value and the operator type
        operand_val = processed_operand.value
        op_type = type(node.op)

        # Only fold plain Python scalars. LLType / Enum instances placed in
        # ast.Constant by visit_Attribute/visit_Name must reach codegen intact,
        # so leave those alone (e.g. do not apply "not"/"~" to a type object).
        if type(operand_val) not in (bool, int, float):
            new_unaryop = create_node_with_attrs(ast.UnaryOp, node, op=node.op, operand=processed_operand)
            return new_unaryop

        # Evaluate the unary operation
        try:
            op_handler = self._unary_handlers.get(op_type)
            if not op_handler:
                # Unknown operator → cannot fold, keep original UnaryOp
                new_unaryop = create_node_with_attrs(ast.UnaryOp, node, op=node.op, operand=processed_operand)
                return new_unaryop

            # Compute the result
            result = op_handler(operand_val)

        except Exception as e:
            # Handle errors (e.g., incompatible type like -"string")
            print(f"Warning: Could not fold UnaryOp {ast.unparse(node)}: {e}")
            new_unaryop = create_node_with_attrs(ast.UnaryOp, node, op=node.op, operand=processed_operand)
            return new_unaryop

        # Return the folded result as a Constant node, preserving location info
        folded_constant = ast.Constant(value=result)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.switch_on_use()
        new_test = self.visit(node.test)
        if isinstance(new_test, ast.Constant):
            if new_test.value:
                ret = self.visit(node.body)
                self.switch_off_use()
                return ret
            else:
                ret = self.visit(node.orelse)
                self.switch_off_use()
                return ret
        new_if = create_node_with_attrs(ast.IfExp, node, test=new_test, body=self.visit(node.body),
                                        orelse=self.visit(node.orelse))
        self.switch_off_use()
        return new_if

    def visit_If(self, node: ast.If) -> ast.AST:
        """Prune statement branches whose test folds to a constant.

        After constexpr substitution a selector (e.g. a derived constexpr
        bool) reads as a literal, so ``if <selector>:`` specializes at the IR
        level instead of surviving as a runtime branch: the dead arm -- and
        every load/store it contains -- is removed before codegen. The kept
        arm is spliced into the enclosing statement list by _process_list.
        """
        self.switch_on_use()
        new_test = self.visit(node.test)
        if isinstance(new_test, ast.Constant):
            kept = node.body if new_test.value else node.orelse
            result = self._process_list(kept)
            self.switch_off_use()
            return result
        new_if = self._copy_node(node)
        new_if.test = new_test
        new_if.body = self._process_list(node.body)
        new_if.orelse = self._process_list(node.orelse)
        self.switch_off_use()
        return new_if

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """
        Process annotated assignments (e.g., "x: const = 5" or "x: int32 = 5").
        """
        # Check if annotation is const (either ConstTypeHelper or ConstAnnotationType)
        is_const_annotation = False
        if node.annotation is not None:
            # Try to resolve the annotation
            try:
                resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx, self.scope_manager)
                # Check if it's const (ConstTypeHelper or ConstAnnotationType)
                if (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                        or resolved_ann == const_annotation):
                    is_const_annotation = True
            except Exception:
                # If resolution fails, check for "const" as a Name node (fallback)
                if isinstance(node.annotation, ast.Name) and node.annotation.id == "const":
                    is_const_annotation = True
                elif isinstance(node.annotation, ast.Attribute) and node.annotation.attr == "const":
                    # Could be ll.const
                    try:
                        resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx, self.scope_manager)
                        if (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                                or resolved_ann == const_annotation):
                            is_const_annotation = True
                    except Exception:
                        pass

        # Process value (with const folding enabled for const annotations)
        self.switch_on_use()
        new_value = self.visit(node.value) if node.value else None
        self.switch_off_use()

        # Process target
        new_target = self.visit(node.target)

        # If it's a const annotation, try to fold the value
        if is_const_annotation:
            # Keep trying to fold until we get a constant
            max_fold_iterations = 10
            for _ in range(max_fold_iterations):
                if isinstance(new_value, ast.Constant):
                    # Successfully folded to constant
                    if isinstance(new_target, ast.Name):
                        # Store in defines for later use
                        self.defines[new_target.id] = new_value.value
                    break
                elif isinstance(new_value, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    # Try to fold again
                    self.switch_on_use()
                    new_value = self.visit(new_value)
                    self.switch_off_use()
                else:
                    # Can't fold further
                    break

        # For const annotations that are successfully folded, we can convert to Assign
        # But for non-const annotations or if we want to preserve AnnAssign for codegen,
        # we should keep it as AnnAssign
        if is_const_annotation:
            if isinstance(new_value, ast.Constant):
                node_copy = self._copy_node(node)
                node_copy.target = new_target
                node_copy.value = new_value
                return node_copy
            else:
                raise_pass_error(f"Can't assign non-const value to const variable: {ast.unparse(node)}", node=node,
                                 ctx=self.ctx)
        else:
            # Non-const type annotation (e.g., "x: uint32 = 5")
            # Preserve the AnnAssign node for codegen to handle
            node_copy = self._copy_node(node)
            node_copy.target = new_target
            node_copy.value = new_value
            node_copy.annotation = node.annotation  # Keep the annotation
            return node_copy

    def visit_Assign(self, node):
        # Skip special struct declarations (e.g., Scheduler(...)) - they should not be const folded
        if hasattr(node, '_is_special_struct_decl') and node._is_special_struct_decl:
            # Preserve the node and its attributes completely
            # However, we still need to visit the constructor arguments to fold constants like Enum values
            # But we need to be careful not to modify the call_node structure itself
            node_copy = self._copy_node(node)
            node_copy.targets = self._process_list(node.targets)
            # Visit the value to fold constants in arguments (e.g., SchedulerGemmType.Normal)
            # This will fold Enum attributes and other constants in the constructor call
            node_copy.value = self.visit(node.value)
            return node_copy

        new_targets = self._process_list(node.targets)
        self.switch_on_use()
        new_value = self.visit(node.value)
        self.switch_off_use()
        reserved_targets = []
        reserved_values = []

        # Check if target has const annotation (e.g., "x: const = 5")
        # Note: This won't happen in regular Assign, only in AnnAssign
        is_const_annotation = False

        if isinstance(new_value, ast.Tuple):
            assert isinstance(new_targets[0], ast.Tuple), type(new_targets[0])
            for t, v in zip(new_targets[0].elts, new_value.elts):
                if not (isinstance(t, ast.Name) and isinstance(v, ast.Name) and v.id == t.id):
                    # avoid self-assignment
                    reserved_targets.append(t)
                    reserved_values.append(v)
                elif isinstance(v, ast.Constant) and not self.in_kernel_function:
                    # Only fold constants if not in kernel function (or if explicitly marked as const)
                    if is_const_annotation:
                        self.defines[t.id] = v.value
            if len(reserved_targets) == 0:
                return None
            reserved_targets = [ast.Tuple(elts=reserved_targets, ctx=new_targets[0].ctx)]
            reserved_values = ast.Tuple(elts=reserved_values, ctx=new_value.ctx)
        else:
            # Check for self-assignment only if target is a Name
            if isinstance(new_value, ast.Name) and isinstance(node.targets[0], ast.Name):
                if new_value.id == node.targets[0].id:
                    # avoid self-assignment
                    return None
            reserved_targets = new_targets
            reserved_values = new_value
            # Only fold constants if:
            # 1. Not in kernel function, OR
            # 2. Explicitly marked with const annotation
            if isinstance(reserved_values, ast.Constant):
                if not self.in_kernel_function:
                    # Global constant
                    if isinstance(reserved_targets[0], ast.Name):
                        self.defines[reserved_targets[0].id] = reserved_values.value
                else:
                    # Kernel variable - it is a local definition
                    # ScopeManager already tracks this, so we don't need kernel_defines logic.
                    pass

        new_assign = create_node_with_attrs(ast.Assign, node, targets=reserved_targets, value=reserved_values)
        return new_assign

    def visit_Attribute(self, node):

        def _recursive_visit_attribute(node):
            if isinstance(node, ast.Attribute):
                val = _recursive_visit_attribute(node.value)
                if isinstance(val, ast.AST):
                    return node
                # otherwise, try to get the attribute
                try:
                    attr = getattr(val, node.attr)
                    if hasattr(attr, BUILTIN_ATTR) and getattr(attr, BUILTIN_ATTR, False):  # builtin
                        # builtin function/class is reserved
                        return node
                    else:
                        return attr
                except AttributeError:
                    # Attribute not found, return node to preserve structure
                    return node
            if isinstance(node, ast.Name):
                # Check scope manager for local shadowing
                if self.scope_manager.is_local(node.id):
                    # It's local, so return AST node (don't resolve from ctx)
                    return node

                if node.id in self.ctx:
                    return self.ctx[node.id]
                # Also check in defines (for constants defined in the code)
                if node.id in self.defines:
                    return self.defines[node.id]
            else:
                return node

        attr = _recursive_visit_attribute(node)
        if isinstance(attr, ast.AST):
            return node
        # Handle Enum instances - they should be folded to ast.Constant
        if isinstance(attr, Enum):
            # Store the Enum instance in ast.Constant
            # The codegen will handle converting it to the appropriate C++ representation
            ret = ast.Constant(value=attr)
            ast.copy_location(ret, node)
            return ret
        # Handle LLType instances - preserve them as LLType objects, not strings
        # This is critical for type inference to work correctly
        if isinstance(attr, LLType):
            ret = ast.Constant(value=attr)
            ast.copy_location(ret, node)
            return ret
        # For other resolved values, convert to ast.Constant
        # But avoid converting LLType to string - this would lose type information
        ret = ast.Constant(value=attr)
        ast.copy_location(ret, node)
        return ret

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.is_rvalue:

            # Check scope manager first
            if self.scope_manager.is_local(node.id):
                # If it's local, we might have folded it (in self.defines)
                if node.id in self.defines:
                    value = self.defines[node.id]
                    if isinstance(value, (int, float, str, LLType)):
                        return ast.Constant(value=value)
                # Otherwise, it's a runtime local variable, return node
                return node

            # Not local, check globals
            if node.id in self.ctx:
                value = self.ctx[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
            elif node.id in self.defines:
                # Global constant defined in this module
                value = self.defines[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
        return node

    def visit_FunctionDef(self, node):
        # Check if this is a kernel function (has @ll_kernel or @lk.ll_kernel decorator)
        is_kernel = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "ll_kernel":
                    is_kernel = True
                    break
                if isinstance(func, ast.Attribute) and func.attr == "ll_kernel":
                    is_kernel = True
                    break
            elif isinstance(decorator, ast.Name) and decorator.id == "ll_kernel":
                is_kernel = True
                break

        # Track if we're in a kernel function
        old_in_kernel = self.in_kernel_function
        if is_kernel:
            self.in_kernel_function = True

        # Scope management
        self.scope_manager.enter_scope()
        self.scope_manager.define_from_ast(node.args)
        # Scan body for locals (Python behavior: assignment anywhere makes it local)
        self.scope_manager.scan_for_locals(node.body)

        new_name = node.name  # Preserve unless mutated
        new_args = self.visit(node.args)
        # Workspace memoization: fold single-definition GMEM cell loads to
        # their stored scalar expression before generic constant folding, so
        # the loaded scalar participates in value analysis with the same
        # documentation as the argument that was smuggled through memory.
        node.body = self._fold_workspace_cell_loads(node)
        new_body = self._process_list(node.body)
        new_decorator_list = self._process_list(node.decorator_list)
        self.switch_on_use()
        new_returns = self.visit(node.returns)
        self.switch_off_use()
        new_type_comment = node.type_comment  # Primitive, no mutation
        new_node = ast.FunctionDef(name=new_name, args=new_args, body=new_body, decorator_list=new_decorator_list,
                                   returns=new_returns, type_comment=new_type_comment)
        ast.copy_location(new_node, node)

        # Exit scope
        self.scope_manager.exit_scope()

        # Restore state
        if is_kernel:
            self.in_kernel_function = old_in_kernel

        return new_node

    def visit_Call(self, node):
        self.switch_on_use()
        # new_func = self.visit(node.func)
        new_args = self._process_list(node.args)
        new_kwargs = self._process_list(node.keywords)
        self.switch_off_use()

        value = recursive_resolve_attribute(node.func, self.ctx, self.scope_manager)
        if isinstance(value, Callable):
            if hasattr(value, CONST_FUNC_ATTR) and getattr(value, CONST_FUNC_ATTR, False):
                for arg in new_args:
                    assert isinstance(
                        arg, ast.Constant
                    ), f"Const function should only take Constant as arguments, but get {arg} {ast.unparse(arg)}"
                call_args = [arg.value for arg in new_args]
                kw_call_args = {}
                for kw in new_kwargs:
                    assert isinstance(kw, ast.keyword)
                    assert isinstance(
                        kw.value,
                        ast.Constant), f"Const function should only take Constant as kw arguments, but get {kw.value}"
                    kw_call_args[kw.arg] = kw.value.value
                ret = value(*call_args, **kw_call_args)
                return ast.Constant(value=ret)
            elif hasattr(value, "__const_type_func__") and value.__const_type_func__:
                assert node in self.all_types, f"Call node {ast.dump(node)} is not in all_types"
                ret_type = self.all_types[node]
                return ast.Constant(value=ret_type)
        # Preserve special attributes (e.g., _is_special_struct_method)
        new_call = create_node_with_attrs(ast.Call, node, func=node.func, args=new_args, keywords=new_kwargs)
        return new_call

    def visit_Assert(self, node):
        new_test = self.visit(node.test)
        new_msg = self.visit(node.msg) if node.msg else None

        # If the test is a constant (after folding), evaluate it at compile time
        if isinstance(new_test, ast.Constant):
            if not new_test.value:
                # Compile-time assert failed
                msg_str = ""
                if new_msg is not None:
                    if isinstance(new_msg, ast.Constant):
                        msg_str = new_msg.value
                    elif isinstance(new_msg, ast.JoinedStr):
                        # Try to evaluate the f-string at compile time
                        try:
                            # Build the string from parts
                            parts = []
                            for val in new_msg.values:
                                if isinstance(val, ast.Constant):
                                    parts.append(str(val.value))
                                elif isinstance(val, ast.FormattedValue):
                                    if isinstance(val.value, ast.Constant):
                                        parts.append(str(val.value.value))
                                    elif isinstance(val.value, ast.Name):
                                        # Check scope
                                        if self.scope_manager.is_local(val.value.id):
                                            pass  # Local
                                        elif val.value.id in self.ctx:
                                            parts.append(str(self.ctx[val.value.id]))
                                        elif val.value.id in self.defines:
                                            parts.append(str(self.defines[val.value.id]))
                                    elif isinstance(val.value, ast.Attribute):
                                        # Try to resolve attribute
                                        try:
                                            attr_val = recursive_resolve_attribute(val.value, self.ctx,
                                                                                   self.scope_manager)
                                            if not isinstance(attr_val, ast.AST):
                                                parts.append(str(attr_val))
                                        except Exception:
                                            pass
                            msg_str = "".join(parts)
                        except Exception:
                            msg_str = ast.unparse(new_msg)
                    else:
                        msg_str = ast.unparse(new_msg) if new_msg else ""

                raise_pass_error(f"Compile-time Assert Failed:\n{msg_str}", node=node, ctx=self.ctx,
                                 error_class=AssertionError)
            # Assert passed at compile time, remove it
            return None

        # Test is not a constant - keep as runtime assert (only allowed in kernel functions)
        if not self.in_kernel_function:
            raise_pass_error(
                f"assert outside kernel should be static (compile-time constant), "
                f"but the test results is {ast.dump(new_test)}", node=node, ctx=self.ctx, error_class=AssertionError)

        # In kernel function, keep the assert as-is (will be codegen'd as runtime assert)
        return ast.Assert(test=new_test, msg=new_msg)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg  # String (parameter name)
        # keep annotation
        node_copy.annotation = self.visit(node.annotation) if node.annotation else None
        node_copy.type_comment = self.visit(node.type_comment)
        return node_copy

    def visit_Subscript(self, node):
        new_value = self.visit(node.value)
        new_slice = self.visit(node.slice)
        if isinstance(new_value, ast.Constant) and isinstance(new_slice, ast.Constant):
            return ast.Constant(value=new_value.value[new_slice.value])
        return ast.Subscript(value=new_value, slice=new_slice, ctx=node.ctx)


def const_fold(tree: ast.AST, ctx: Dict[str, Any] = None) -> ast.AST:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    folder = ConstantFolder(ctx=ctx, all_types=all_types)
    # Try to get source lines for error reporting
    folder.source_lines = get_source_from_ctx(ctx)
    return ast.fix_missing_locations(folder.visit(tree))
