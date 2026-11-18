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
"""CPU regression for workspace-cell scalar memoization in const folding.

Capability ``ir.analysis.const_fold_cross_tree_scalar_memoization``:

a load from a GMEM workspace cell whose only definition in the enclosing
function is a single store of a scalar expression resolves to that stored
expression, so the scalar's documented value (e.g. an incoming kernel
argument that a frontend smuggled through memory, ``kvc[0] = kv0``)
propagates into the reading expression tree (``_kv = kvc[b]`` with ``b`` a
constant) instead of collapsing to an opaque runtime load.

These tests exercise the shared :class:`ConstantFolder` pass only — no GPU,
no CUDA toolchain.  Folding fires exactly when the store is a top-level
statement strictly dominating the load, the index is provably the same cell
on both sides (a plain integer literal, or a local bound exactly once to an
integer literal such as ``b: const = 1``), the defining store is unique for
that cell, the buffer base never escapes through a call argument, the stored
expression is duplicable (names/constants and arithmetic — never a ``Call``
or a memory read, since the store is kept and a copy is spliced into each
load), and the stored expression's free names are never rebound between store
and load.  Every relaxation below must *not* fold: folding there would
mis-document a value the runtime can disagree with.
"""

import ast
import unittest

from little_kernel.core.passes.constfold import ConstantFolder


def _fold(src: str) -> ast.FunctionDef:
    tree = ast.parse(src)
    folder = ConstantFolder(ctx={}, all_types={})
    out = ast.fix_missing_locations(folder.visit(tree))
    return out.body[0]


def _rhs_of_assign(func: ast.FunctionDef, target: str) -> str:
    """Unparsed RHS of the top-level ``target = ...`` assignment, or None."""
    for stmt in func.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == target):
            return ast.unparse(stmt.value)
    return None


class TestWorkspaceCellMemoization(unittest.TestCase):

    def test_single_store_load_folds_to_stored_scalar(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    _kv = kvc[0]
    nt = (_kv + 127) // 128
    for jt in range(nt):
        pass
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_const_name_load_index_folds(self):
        # Headline case: the load index is a compile-time-constant *name*
        # (``b: const = 1``), not a literal.  Resolution anticipates the
        # generic fold that rewrites ``b`` to ``1``; both sides name cell 1.
        func = _fold("""
def entry(kv0, kvc):
    b: const = 1
    kvc[1] = kv0
    _kv = kvc[b]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_const_name_index_on_store_and_load_folds(self):
        # Both the store and the load use the const-bound name.
        func = _fold("""
def entry(kv0, kvc):
    b: const = 1
    kvc[b] = kv0
    _kv = kvc[b]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_plain_literal_const_name_in_wrong_cell_does_not_fold(self):
        # ``b = 1`` bound to 1, but the store used cell 0: no match.  The
        # load must not resolve to the stored scalar ``kv0``; generic const
        # folding may rewrite the index name to the literal ``1`` — that is
        # the same conservatism the memoization anticipates — so assert the
        # loaded value is still a subscript read of ``kvc``, not ``kv0``.
        func = _fold("""
def entry(kv0, kvc):
    b = 1
    kvc[0] = kv0
    _kv = kvc[b]
""")
        rhs = _rhs_of_assign(func, "_kv")
        self.assertNotEqual(rhs, "kv0")
        self.assertIn(rhs, ("kvc[b]", "kvc[1]"))

    def test_rebound_index_name_is_dynamic(self):
        # ``b`` bound twice: no longer a provable single cell index.
        func = _fold("""
def entry(kv0, kvc):
    b = 1
    kvc[b] = kv0
    b = 2
    _kv = kvc[1]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[1]")

    def test_stored_call_not_duplicated(self):
        # Purity: the defining store is kept, so splicing the stored
        # expression into the load would evaluate ``next_val()`` twice (a
        # miscompile for impure callees).  Cells storing a Call never fold.
        func = _fold("""
def entry(kvc):
    kvc[0] = next_val()
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")
        first = func.body[0]
        self.assertEqual(ast.unparse(first.value), "next_val()")

    def test_stored_subscript_read_not_duplicated(self):
        # A stored *memory read* is not duplicable documentation either.
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = other[0]
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_distinct_cells_do_not_fold(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    _kv = kvc[1]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[1]")

    def test_double_store_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kvc[0] = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_dynamic_load_index_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc, j):
    kvc[0] = kv0
    _kv = kvc[j]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[j]")

    def test_dynamic_store_somewhere_else_disqualifies_base(self):
        func = _fold("""
def entry(kv0, kvc, j):
    kvc[j] = 7
    kvc[0] = kv0
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_call_escaped_base_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    may_mutate(kvc)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_starred_call_escaped_base_does_not_fold(self):
        # ``may_mutate(*kvc)`` splices the buffer's elements into the callee's
        # positional arguments, handing the buffer to arbitrary code exactly
        # like the plain whole-name form ``may_mutate(kvc)`` does.  The parser
        # wraps a starred pass-through as Starred(value=Name(kvc)) in
        # ``node.args`` — never a bare Name — so an argument scan that only
        # matches ``ast.Name`` misses the escape and folds the later load to
        # the stale pre-call stored value (a silent miscompile).
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    may_mutate(*kvc)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_double_starred_call_escaped_base_does_not_fold(self):
        # The keyword-unpacking form ``may_mutate(**kvc)`` carries the buffer
        # as an ``ast.keyword(arg=None, value=Name(kvc))``; the existing
        # keyword scan already treats it as a base escape.  Pin that behavior
        # so the starred-argument fix does not regress the mirrored path.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    may_mutate(**kvc)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_starred_call_of_other_base_still_folds_this_base(self):
        # The escape check is per base, not per statement: starring an
        # *unrelated* buffer must not disqualify the singly-defined ``kvc``
        # cell.  This guards against an over-broad fix that star-marks every
        # base or the whole function.
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = kv0
    may_mutate(*other)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_aliased_base_store_does_not_fold(self):
        # The minimum reviewer-required control: ``alias = kvc`` creates a
        # second name for the buffer, so a store through the alias
        # (``alias[0] = 9``) is recorded under ``alias`` and the
        # single-definition proof for ``kvc`` must be voided — exactly like
        # the whole-buffer escape ``may_mutate(kvc)`` already was.  Without
        # the aliasing rule this folds ``_kv`` to the stale ``kv0`` while the
        # runtime cell holds 9.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    alias = kvc
    alias[0] = 9
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_aliased_base_dynamic_store_does_not_fold(self):
        # The dynamic-store variant from the review: the alias is written
        # with a non-constant index, so even the cell it may have written is
        # unknowable — the whole base must be disqualified.
        func = _fold("""
def entry(kv0, kvc, j):
    kvc[0] = kv0
    alias = kvc
    alias[j] = 9
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_tuple_target_aliasing_does_not_fold(self):
        # Sequence-unpacking binds the alias from the same read:
        # ``alias, z = kvc, 1`` has ``kvc`` as a plain RHS Name load inside a
        # Tuple, so the escape scan must catch it without a dedicated
        # statement-shape case.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    alias, z = kvc, 1
    alias[0] = 9
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_for_target_aliasing_does_not_fold(self):
        # The alias binding can also sit in a loop target over a literal
        # tuple holding the base — one iteration, but a real aliasing write.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    for alias in (kvc,):
        alias[0] = 9
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_aliasing_other_base_still_folds_this_base(self):
        # The escape rule is per base: aliasing an *unrelated* buffer must
        # not disqualify the singly-defined ``kvc`` cell.  Guards against an
        # over-broad fix that poisons every base on any untracked Name load.
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = kv0
    alias = other
    alias[0] = 9
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_stored_expression_name_reads_still_fold(self):
        # And the symmetric positive guard for the escape scan itself: pure
        # Name loads *inside the stored scalar expression* (``tid + kv0``)
        # are the duplicable documentation this pass exists to propagate, so
        # they must remain foldable — only whole reads of the *cell base*
        # disqualify.
        func = _fold("""
def entry(tid, kv0, kvc):
    kvc[0] = tid + kv0
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "tid + kv0")

    def test_rebound_stored_name_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kv0 = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_guarded_store_does_not_fold(self):
        # A store inside ``if`` does not dominate top-level loads without a
        # barrier-domination proof; const folding stays conservative here.
        func = _fold("""
def entry(tid, kv0, kvc):
    if tid == 0:
        kvc[0] = kv0
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_nested_load_does_not_fold(self):
        # Loads inside control flow are not positionally dominated proofs.
        func = _fold("""
def entry(tid, kv0, kvc):
    kvc[0] = kv0
    if tid > 3:
        _kv = kvc[0]
""")
        nested = func.body[1].body[0]
        self.assertEqual(ast.unparse(nested.value), "kvc[0]")

    def test_augassign_to_base_disqualifies(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kvc[0] += 3
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_stored_expression_shape_preserved(self):
        # The fold substitutes the stored *expression* verbatim (not only
        # plain names), letting later trees keep the full documentation.
        func = _fold("""
def entry(kv0, kvc):
    kvc[2] = (kv0 + 1) * 2
    _kv = kvc[2]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "(kv0 + 1) * 2")

    def test_store_itself_is_kept(self):
        # Folding rewrites the load, never the defining store: the memory
        # side effect must remain in the emitted kernel.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    _kv = kvc[0]
""")
        self.assertEqual(len(func.body), 2)
        first = func.body[0]
        self.assertIsInstance(first, ast.Assign)
        self.assertEqual(ast.unparse(first.targets[0]), "kvc[0]")

    def test_earlier_use_not_folded(self):
        # A load before the unique store must not see the later value.
        func = _fold("""
def entry(kv0, kvc):
    early = kvc[0]
    kvc[0] = kv0
    late = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "early"), "kvc[0]")
        self.assertEqual(_rhs_of_assign(func, "late"), "kv0")

    def test_other_bases_unaffected(self):
        func = _fold("""
def entry(kv0, kv1, kvc, other):
    kvc[0] = kv0
    _a = kvc[0]
    _b = other[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_a"), "kv0")
        self.assertEqual(_rhs_of_assign(func, "_b"), "other[0]")

    def test_base_name_reassigned_does_not_fold(self):
        # After ``kvc = other`` the load reads ``other[0]``; folding it to
        # the value stored into the old buffer would be a miscompile.
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = kv0
    kvc = other
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_base_name_reassigned_via_tuple_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = kv0
    kvc, _z = other, 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_base_name_reassigned_via_walrus_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc, other):
    kvc[0] = kv0
    n = len(kvc := other)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_base_name_plain_augassign_does_not_fold(self):
        # ``kvc += delta`` rebinds the base name (Name target, not a
        # subscript read-modify-write), so the base is no longer stable.
        func = _fold("""
def entry(kv0, kvc, delta):
    kvc[0] = kv0
    kvc += delta
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_import_rebinds_free_name_of_stored_expr(self):
        # ``from x import m`` binds ``m`` via ast.alias, not Name(Store);
        # the stored expression's ``m`` is stale after the import.
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    from x import m
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_plain_import_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    import m
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_import_alias_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    import x as m
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_import_disqualifies_const_index_name(self):
        # ``b`` was bound once to a literal, but the import rebinds it, so
        # it is no longer a provable cell index for the load.
        func = _fold("""
def entry(kv0, kvc):
    b = 1
    kvc[1] = kv0
    import x as b
    _kv = kvc[b]
""")
        self.assertNotEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_global_declaration_disqualifies_base(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    global kvc
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_nested_def_rebinds_base_name(self):
        # ``def kvc(): ...`` rebinds the local ``kvc`` to a function object
        # without any Name(Store) node; folding the later load to the value
        # stored in the old buffer would be a miscompile.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    def kvc(): pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_nested_async_def_rebinds_base_name(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    async def kvc(): pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_class_def_rebinds_base_name(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    class kvc: pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_except_as_rebinds_free_name_of_stored_expr(self):
        # ``except E as m:`` binds ``m`` via ExceptHandler.name, not a
        # Name(Store); the stored expression's ``m`` is stale past it.  The
        # base itself is untouched, so the positional invalidation (not the
        # fact pass) is what must catch this.
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    try:
        pass
    except E as m:
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_match_capture_rebinds_base_name(self):
        # ``case kvc:`` binds ``kvc`` through a MatchAs pattern without any
        # Name(Store) node; the later load no longer reads the old buffer.
        func = _fold("""
def entry(kv0, kvc, tag):
    kvc[0] = kv0
    match tag:
        case kvc:
            pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_match_star_capture_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m, seq):
    kvc[0] = kv0 + m
    match seq:
        case [*m]:
            pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_match_mapping_rest_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m, d):
    kvc[0] = kv0 + m
    match d:
        case {"k": 1, **m}:
            pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_for_target_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    for m in (1, 2):
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_with_as_rebinds_free_name_of_stored_expr(self):
        func = _fold("""
def entry(kv0, kvc, m, res):
    kvc[0] = kv0 + m
    with res as m:
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_del_free_name_disqualifies_fold(self):
        # ``del m`` (Name with Del context) removes the stored expression's
        # free name; the load must stay a runtime read.
        func = _fold("""
def entry(kv0, kvc, m):
    kvc[0] = kv0 + m
    del m
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_walrus_rebinds_free_name_in_same_statement_as_load(self):
        # Same-statement rebinding: Python evaluates the RHS left-to-right,
        # so ``(x := 5)`` rebinds ``x`` *before* ``kvc[0]`` executes — while
        # the store the fold would substitute from saw the *old* ``x``.
        # Statement-granular invalidation that runs only after rewriting the
        # statement would mis-fold the load to the new ``x``; the check must
        # run against the statement's own bound names *before* folding it.
        func = _fold("""
def entry(x, n, kvc):
    kvc[0] = x
    n = (x := 5) + kvc[0]
""")
        rhs = _rhs_of_assign(func, "n")
        self.assertNotEqual(rhs, "(x := 5) + x")
        self.assertEqual(rhs, "(x := 5) + kvc[0]")

    def test_method_call_on_base_does_not_fold(self):
        # ``kvc.fill(0)`` hands the buffer to the callee as the implicit
        # first argument (``Call.func`` is ``Attribute(value=Name kvc)``,
        # never a plain-name argument); it may mutate the buffer in place,
        # the same aliasing hazard as ``may_mutate(kvc)``.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kvc.fill(0)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_chained_store_is_a_second_definition(self):
        # ``a = kvc[0] = kv0 + 1`` stores the cell through a second Assign
        # target; the single-target filter must not hide it from the
        # single-definition proof.  The runtime cell value is ``kv0 + 1``,
        # so folding the later load to the first store's ``kv0`` would be a
        # silent miscompile.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    a = kvc[0] = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_annotated_store_is_a_second_definition(self):
        # ``kvc[0]: int = other`` (ast.AnnAssign with a Subscript target)
        # never reaches visit_Assign; it must still count as a cell
        # definition.  The runtime cell value is ``other``.
        func = _fold("""
def entry(kv0, other, kvc):
    kvc[0] = kv0
    kvc[0]: int = other
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_del_cell_is_a_definition_event(self):
        # ``del kvc[0]`` invalidates the cell; a later load must not resolve
        # to the pre-``del`` stored value.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    del kvc[0]
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_second_store_inside_for_does_not_fold(self):
        # A store nested in a ``for`` body executes between the top-level
        # store and the later top-level load; the load must not fold to the
        # stale first-store value.
        func = _fold("""
def entry(kv0, kvc, n):
    kvc[0] = kv0
    for i in range(n):
        kvc[0] = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_second_store_inside_if_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc, c):
    kvc[0] = kv0
    if c > 0:
        kvc[0] = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_second_store_inside_while_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc, n):
    kvc[0] = kv0
    while n > 0:
        kvc[0] = kv0 + 1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_second_store_inside_try_does_not_fold(self):
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    try:
        kvc[0] = kv0 + 1
    except E:
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_nested_store_of_other_cell_still_folds_this_cell(self):
        # The cross-check is per cell, not per base: a nested store to cell
        # 1 must not poison the fold of the singly-defined cell 0.
        func = _fold("""
def entry(kv0, kv1, kvc, n):
    kvc[0] = kv0
    for i in range(n):
        kvc[1] = kv1
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")

    def test_top_level_and_nested_store_of_same_cell_from_dynamic_guard(self):
        # Combined guard: only one top-level store exists, but the cell has
        # a second definition reachable at runtime (guarded loop).  After
        # the fix the cell's recorded definitions no longer match the
        # modeled top-level count, so the load stays a runtime read.
        func = _fold("""
def entry(kv0, kvc, flag):
    kvc[0] = kv0
    if flag:
        kvc[0] = 7
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_tuple_target_store_is_a_second_definition(self):
        # ``kvc[0], a = kv0 + 1, 0`` stores the cell through a Tuple element;
        # the Subscript carries a Store context even nested in the target, so
        # the single-definition proof must count it.  The runtime cell value
        # is ``kv0 + 1``; folding the later load to the first store's ``kv0``
        # would be a silent miscompile.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kvc[0], a = kv0 + 1, 0
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_starred_tuple_target_store_is_a_second_definition(self):
        # ``kvc[0], *rest = vals`` binds the cell to ``vals[0]`` at runtime.
        func = _fold("""
def entry(kv0, kvc, vals):
    kvc[0] = kv0
    kvc[0], *rest = vals
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_list_target_store_is_a_second_definition(self):
        # ``[kvc[0], *rest] = vals`` is an ast.List target (not ast.Tuple);
        # the same Store-context accounting must see it.
        func = _fold("""
def entry(kv0, kvc, vals):
    kvc[0] = kv0
    [kvc[0], *rest] = vals
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_for_target_subscript_store_is_a_definition(self):
        # ``for kvc[0] in it:`` writes the cell through the loop target — a
        # Store-context Subscript that no Assign/Delete visitor can see.
        # The runtime cell value after the loop differs from ``kv0``.
        func = _fold("""
def entry(kv0, kvc, it):
    kvc[0] = kv0
    for kvc[0] in it:
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_with_as_subscript_store_is_a_definition(self):
        # ``with r as kvc[0]:`` writes the cell through the with-item
        # optional target, again a Store-context Subscript.
        func = _fold("""
def entry(kv0, kvc, r):
    kvc[0] = kv0
    with r as kvc[0]:
        pass
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_tuple_del_is_a_definition_event(self):
        # ``del (kvc[0], a)`` carries the subscript in Del context inside a
        # Tuple del-target; the pre-``del`` stored value must not fold.
        func = _fold("""
def entry(kv0, kvc, a):
    kvc[0] = kv0
    del (kvc[0], a)
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")

    def test_augassign_store_on_cell_disqualifies(self):
        # ``kvc[0] += 3`` rewrites cell 0 with a Store-context subscript; a
        # still-earlier load must not fold to the stale pre-AugAssign value.
        # (Sibling of test_augassign_to_base_disqualifies, which covers the
        # plain-Name ``kvc += delta`` form that rebinds the base itself.)
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kv0")
        # Now with the intervening in-place add on the same cell the fold
        # must not apply — this is the exact reviewer reproduction shape.
        func = _fold("""
def entry(kv0, kvc):
    kvc[0] = kv0
    kvc[0] += 3
    _kv = kvc[0]
""")
        self.assertEqual(_rhs_of_assign(func, "_kv"), "kvc[0]")


if __name__ == "__main__":
    unittest.main()
