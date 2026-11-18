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
"""Divisibility (multiple-of) analysis for vector ld/st legality queries.

Pure-CPU regression coverage for
``little_kernel.codegen.visitors.alignment_analysis``: the shared,
frontend-agnostic analysis that decides when a DYNAMIC integer offset
(gather row base, thread-tile base, linearized index) is provably a
multiple of a vector width, the alignment precondition every
``ld/st_global_v4_*`` intrinsic states as a legality contract.

Under test:
  * fact propagation through +, -, *, //, %, << with C-truncating semantics;
  * conservative None (unknown) for unanalyzable expressions, never a wrong
    fact;
  * the query API (proves_multiple_of / proves_congruent) used by a
    vector-legalization decision, including the literal-0 short-circuit;
  * composition with compute_static_int_consts via env_from_literal_bindings;
  * the motivating gather pattern: (t * E + e) * P with P a multiple of the
    vector width is provably aligned for every dynamic t, E, e.
"""

import ast
import unittest

from little_kernel.codegen.visitors.alignment_analysis import analyze, env_from_literal_bindings, literal_fact, proves_congruent, proves_multiple_of
from little_kernel.codegen.visitors.static_int_analysis import compute_static_int_consts


def _expr(src: str) -> ast.AST:
    """Parse a one-line integer expression source into its AST node."""
    return ast.parse(src, mode="eval").body


class TestFactPropagation(unittest.TestCase):

    def test_literal_has_itself_as_gcd(self):
        self.assertEqual(literal_fact(12), (12, 0))
        self.assertEqual(literal_fact(-8), (8, 0))
        # Zero is special: no finite gcd encodes it; analysis of the
        # literal 0 must still answer divisibility queries exactly.
        self.assertEqual(literal_fact(0), (1, 0))

    def test_literal_addition_combines_gcds(self):
        # 12 + 8: both multiples of 4; sum provably multiple of 4, and the
        # exact value folds (literal env), so gcd stays tight at 20.
        fact = analyze(_expr("12 + 8"), {})
        self.assertIsNotNone(fact)
        g, r = fact
        self.assertEqual(g % 4, 0)
        self.assertEqual(r, 0)

    def test_name_times_literal_keeps_literal_divisors(self):
        # idx unknown, (4,0) environment fact: idx * 8 provably mult of 8*4.
        fact = analyze(_expr("idx * 8"), {"idx": (4, 0)})
        self.assertEqual(fact, (32, 0))

    def test_literal_times_unknown_name_yields_literal_gcd(self):
        # No env fact for idx: idx * 4 provably multiple of 4 (the generic
        # "any integer times k is a multiple of k" case, gcd comes from the
        # literal alone).
        fact = analyze(_expr("idx * 4"), {})
        self.assertEqual(fact, (4, 0))

    def test_sum_of_aligned_terms_stays_aligned(self):
        # a: mult of 16, b: mult of 8 -> a + b provably mult of gcd(16,8)=8
        fact = analyze(_expr("a + b"), {"a": (16, 0), "b": (8, 0)})
        self.assertEqual(fact, (8, 0))

    def test_residue_tracks_through_addition(self):
        # a = mult of 8, c = mult of 4 plus 2 -> a + c is 2 mod 4.
        fact = analyze(_expr("a + c"), {"a": (8, 0), "c": (4, 2)})
        self.assertEqual(fact, (4, 2))

    def test_subtraction_residue(self):
        fact = analyze(_expr("a - c"), {"a": (8, 0), "c": (4, 2)})
        # -2 mod 4 == 2
        self.assertEqual(fact, (4, 2))

    def test_mul_with_two_nonzero_residues_stays_sound(self):
        # (4k+2) * (4m+2) = 16km + 8k + 8m + 4: provably 4 mod 8, never a
        # multiple of 8; the analysis must not inflate the modulus to 16.
        fact = analyze(_expr("a * b"), {"a": (4, 2), "b": (4, 2)})
        g, r = fact
        self.assertNotEqual(g % 16, 0)
        self.assertTrue(proves_multiple_of(_expr("a * b"), 4, {"a": (4, 2), "b": (4, 2)}))
        self.assertFalse(proves_multiple_of(_expr("a * b"), 8, {"a": (4, 2), "b": (4, 2)}))
        # Concrete witness: residues hold for the smallest members 2*2=4.
        self.assertEqual(4 % g, r)

    def test_division_by_constant_keeps_scaled_divisors(self):
        # a mult of 16; a // 4 provably mult of 4.
        fact = analyze(_expr("a // 4"), {"a": (16, 0)})
        self.assertEqual(fact, (4, 0))

    def test_division_by_nondivisor_degrades_to_universal(self):
        # a mult of 6; a // 4: nothing provable about the quotient beyond
        # the vacuous 0 mod 1 fact.
        self.assertEqual(analyze(_expr("a // 4"), {"a": (6, 0)}), (1, 0))

    def test_division_by_name_degrades_to_universal(self):
        self.assertEqual(analyze(_expr("a // b"), {"a": (16, 0)}), (1, 0))

    def test_modulo_by_constant_gives_gcd(self):
        # (a mult of 12) % 16: result mult of gcd(12,16)=4.
        fact = analyze(_expr("a % 16"), {"a": (12, 0)})
        self.assertEqual(fact, (4, 0))

    def test_left_shift_by_constant_scales(self):
        fact = analyze(_expr("a << 2"), {"a": (3, 0)})
        self.assertEqual(fact, (12, 0))

    def test_right_shift_degrades_to_universal(self):
        self.assertEqual(analyze(_expr("a >> 2"), {"a": (16, 0)}), (1, 0))

    def test_unary_minus_preserves_gcd(self):
        fact = analyze(_expr("-a"), {"a": (8, 2)})
        self.assertEqual(fact, (8, 6))  # -2 mod 8

    def test_unknown_leaf_in_context_keeps_only_universal(self):
        # idx unknown: idx + 4 carries no information beyond the trivial
        # 0 mod 1 fact; crucially it NEVER reports a wrong divisor.
        self.assertEqual(analyze(_expr("idx + 4"), {}), (1, 0))
        self.assertFalse(proves_multiple_of(_expr("idx + 4"), 4, {}))
        self.assertFalse(proves_multiple_of(_expr("idx + 4"), 2, {}))

    def test_bare_unknown_name_is_none(self):
        # A lone opaque node has no analyzable fragment at all: None.
        self.assertIsNone(analyze(_expr("idx"), {}))

    def test_unsupported_leaf_nodes_are_none(self):
        for src in ("f(x)", "x.y", "arr[0]", "1.5"):
            self.assertIsNone(analyze(_expr(src), {"x": (4, 0)}), msg=src)

    def test_unsupported_ops_degrade_to_universal(self):
        for src in ("x ** 2", "x | 3", "x & 3"):
            self.assertEqual(analyze(_expr(src), {"x": (4, 0)}), (1, 0), msg=src)
            self.assertFalse(proves_multiple_of(_expr(src), 2, {"x": (4, 0)}), msg=src)


class TestQueryApi(unittest.TestCase):

    def test_proves_multiple_of_positive_cases(self):
        env = {"a": (16, 0), "b": (8, 0), "c": (4, 0)}
        self.assertTrue(proves_multiple_of(_expr("a + b"), 8, env))
        self.assertTrue(proves_multiple_of(_expr("a + c"), 4, env))
        self.assertTrue(proves_multiple_of(_expr("a * 3"), 16, env))

    def test_proves_multiple_of_negative_cases(self):
        env = {"a": (4, 2)}  # 2 mod 4: NOT aligned
        self.assertFalse(proves_multiple_of(_expr("a"), 4, env))
        self.assertFalse(proves_multiple_of(_expr("unknown"), 4, env))
        # 4 mod 8 proves mult of 4 but not mult of 8
        self.assertTrue(proves_multiple_of(_expr("c"), 4, {"c": (8, 4)}))
        self.assertFalse(proves_multiple_of(_expr("c"), 8, {"c": (8, 4)}))

    def test_proves_multiple_of_literal_zero(self):
        self.assertTrue(proves_multiple_of(_expr("0"), 16, {}))
        self.assertFalse(proves_multiple_of(_expr("17"), 16, {}))

    def test_proves_congruent(self):
        env = {"a": (16, 8)}  # a == 8 mod 16: a v2 access at 8-byte phase
        self.assertTrue(proves_congruent(_expr("a"), 16, 8, env))
        self.assertTrue(proves_congruent(_expr("a"), 8, 0, env))
        self.assertFalse(proves_congruent(_expr("a"), 16, 0, env))
        self.assertFalse(proves_congruent(_expr("unknown"), 16, 8, env))

    def test_bad_modulus_raises(self):
        for k in (0, -4):
            with self.assertRaises(ValueError):
                proves_multiple_of(_expr("4"), k, {})
            with self.assertRaises(ValueError):
                proves_congruent(_expr("4"), k, 0, {})


class TestGatherLayoutScenario(unittest.TestCase):
    """The audit's motivating pattern: dynamic row-gather offsets into a
    packed [S, P] int32 table, with vector width 4 (16 B / 4 B)."""

    def test_row_base_with_stride_multiple_of_vector_width(self):
        # sf row-gather with P = 8 packs per row, flattened vector index
        # derived as r * (P // 4) = r * 2, element offset = (r * 2) * 4:
        # provably a multiple of 4 for every dynamic r.
        off = _expr("(r * 2) * 4")
        self.assertTrue(proves_multiple_of(off, 4, {}))

    def test_linearized_index_times_pack_stride(self):
        # data row index (t * E + e) with dynamic t, E, e, then * 4 packs:
        # linearization loses nothing; final offset provably mult of 4.
        off = _expr("(t * E + e) * 4")
        self.assertTrue(proves_multiple_of(off, 4, {}))
        self.assertFalse(proves_multiple_of(_expr("t * E + e"), 4, {}))

    def test_ragged_lane_offset_breaks_proof(self):
        # idx * 4 + 2: provably 2 mod 4 -> vector load at this element
        # offset is provable NOT aligned, so the scalar path is mandatory.
        off = _expr("idx * 4 + 2")
        self.assertFalse(proves_multiple_of(off, 4, {}))
        self.assertTrue(proves_congruent(off, 4, 2, {}))

    def test_static_stride_bindings_propagate(self):
        # A frontend pins P: ll.int32 = 8 as a static local; composing
        # compute_static_int_consts with env_from_literal_bindings must let
        # idx * P + 0 prove alignment without special-casing.
        func = ast.parse("def k(idx):\n"
                         "    P: int = 8\n"
                         "    pass\n").body[0]
        statics = compute_static_int_consts(
            func,
            resolve_annotation=lambda _ann: __import__("little_kernel.core.type_system", fromlist=["int32"]).int32,
            infer_expr_type=lambda _n, _scope: __import__("little_kernel.core.type_system", fromlist=["int32"]).int32)
        env = env_from_literal_bindings({name: rec.value for name, rec in statics.items()})
        self.assertTrue(proves_multiple_of(_expr("idx * P"), 4, env))
        self.assertTrue(proves_multiple_of(_expr("idx * P"), 8, env))

    def test_exact_literal_fold_in_expression(self):
        # 12 // 4 folds only through literal facts: provably mult of 3.
        self.assertTrue(proves_multiple_of(_expr("12 // 4"), 3, {}))
        self.assertFalse(proves_multiple_of(_expr("12 // 4"), 4, {}))


class TestExhaustiveSoundness(unittest.TestCase):
    """Brute-force witness check: for every operator, every small fact pair,
    and every concrete value consistent with the input facts, the reported
    output fact must hold for the computed value.  A violation here would be
    an invented divisor — an illegal vectorization waiting to happen."""

    def _check_binary(self, op: str, c_fn):
        env = {"a": None, "b": None}
        facts = [(g, r) for g in (1, 2, 3, 4, 6, 8, 12, 16) for r in range(g)]
        for fa in facts:
            for fb in facts:
                expr = _expr(f"a {op} b")
                for ka in range(-2, 3):
                    va = ka * fa[0] + fa[1]
                    for kb in range(-2, 3):
                        vb_ = kb * fb[0] + fb[1]
                        env["a"], env["b"] = fa, fb
                        # Only test fact-soundness when the op is defined.
                        if op in ("//", "%") and vb_ == 0:
                            continue
                        got = analyze(expr, dict(env))
                        self.assertIsNotNone(got, (op, fa, fb))
                        g, r = got
                        try:
                            val = c_fn(va, vb_)
                        except ZeroDivisionError:
                            continue
                        self.assertEqual(val % g, r, f"{va} {op} {vb_} = {val} violates "
                                         f"({g},{r}) from {fa} {op} {fb}")

    @staticmethod
    def _c_div(a, b):
        q = abs(a) // abs(b)
        return -q if (a < 0) != (b < 0) else q

    def test_add_sound(self):
        self._check_binary("+", lambda a, b: a + b)

    def test_sub_sound(self):
        self._check_binary("-", lambda a, b: a - b)

    def test_mul_sound(self):
        self._check_binary("*", lambda a, b: a * b)

    def test_floordiv_sound(self):
        self._check_binary("//", self._c_div)

    def test_mod_sound(self):
        self._check_binary("%", lambda a, b: a - self._c_div(a, b) * b)

    def test_shift_and_unary_minus_sound(self):
        facts = [(g, r) for g in (1, 2, 4, 8, 16) for r in range(g)]
        for fa in facts:
            for shift in range(0, 6):
                got = analyze(_expr("a << s"), {"a": fa})
                # Non-constant shift amount: universal fact, trivially sound.
                self.assertEqual(got, (1, 0))
                got = analyze(_expr(f"a << {shift}"), {"a": fa})
                g, r = got
                for k in range(-2, 3):
                    val = (k * fa[0] + fa[1]) << shift
                    self.assertEqual(val % g, r)
            got = analyze(_expr("-a"), {"a": fa})
            g, r = got
            for k in range(-2, 3):
                self.assertEqual((-1 * (k * fa[0] + fa[1])) % g, r)


if __name__ == "__main__":
    unittest.main()
