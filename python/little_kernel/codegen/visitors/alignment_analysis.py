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
# under the following conditions:
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
"""Provable divisibility ("multiple-of") analysis for integer expressions.

Every 16-byte global/shared vector access intrinsic in
``language/intrin/memory.py`` states the same legality precondition in its
docstring: the byte address ``base + elem_offset * sizeof(dtype)`` must be
16-byte (or 8-byte, for the v2 forms) aligned, and a misaligned address is a
hardware fault.  The compiler, however, had no shared way to *reason* about
that precondition for DYNAMIC offsets: frontends either hard-code trust in a
layout or fall back to scalar lanes everywhere, even when alignment is
provable by modular arithmetic.  Typical provable cases:

* a gather whose row stride is a multiple of the vector width in elements
  (e.g. offset ``idx * P + p`` with ``P % 4 == 0`` and ``p // 4`` the
  vector index: the element offset ``idx * P + (p - p % 4)`` is a multiple
  of 4);
* a thread tile base ``tid * TILE`` with ``TILE`` a multiple of 4;
* a dimension size multiplied down an index linearization
  (``(t * E + e) * P`` keeps every factor's divisors).

This module provides a small, conservative, shared evaluator that tracks
integer divisibility of source-level integer expressions, in the same style
and with the same soundness discipline as ``static_int_analysis``:

* every name maps to a ``DivisibilityFact`` = (gcd, residue) meaning
  ``name % gcd == residue`` whenever the fact was proven;
* integer literals map to their own value as an exact residue at any
  modulus (they are their own gcd and residue);
* binary ``+ -`` combine gcds via gcd(g1, g2) and residues modulo the
  combined gcd; ``*`` combines via gcd(g1*g2_abs...)
  by exact rules below; ``//`` and ``%`` only propagate when the divisor is
  a known constant and standard modular rules apply, otherwise they drop
  the fact (conservative);
* shifts combine under exact constant semantics;
* anything outside the supported fragment (calls, attributes, subscripts,
  unknown names, comparisons, floats) yields no fact, which the caller
  must treat as "legality unknown" and keep the scalar path.

Soundness contract: a reported fact is *necessary* — if ``analyze`` reports
``g`` divides ``expr - r``, then every concrete evaluation of the
expression under the environment facts satisfies ``expr % g == r`` with
C-style truncating semantics.  A missing fact NEVER means "not divisible";
it only means "not proven".  Callers decide legality with a single query
such as :func:`proves_multiple_of`, which returns True only when the
analysis derived a fact with ``gcd % k == 0 and residue % k == 0``.

The analysis deliberately does NOT read data-dependent values (memory
contents), does NOT assume tensor base addresses' alignment (that is a
toolchain property the frontend must assert separately), and does NOT emit
code — it is a pure query over an ``ast`` expression plus an environment,
usable from type inference, from a pass validating a vectorized rewrite, or
from a frontend deciding between vector and scalar paths.
"""

import ast
from math import gcd as _math_gcd
from typing import Dict, Mapping, Optional, Tuple

# A divisibility fact: value % gcd == residue, with gcd > 0 and
# 0 <= residue < gcd (residue is canonicalized with Python floor-mod —
# remainder CLASS is what alignment needs, C truncation only fixes the
# representative, which canonicalization subsumes).
DivisibilityFact = Tuple[int, int]  # (gcd, residue mod gcd)


def _canon(g: int, r: int) -> DivisibilityFact:
    """Canonical fact form: g >= 1, 0 <= r < g."""
    g = abs(g)
    if g == 0:
        # gcd 0 means "exactly known value 0"; represent as modulus 1.
        return (1, 0)
    return (g, r % g)


def literal_fact(value: int) -> DivisibilityFact:
    """Fact for a known integer literal: exact under modulus |value|+.

    Uses gcd = |value| so any divisor of the literal is provable; the
    residue is 0.  Zero is special: 0 is a multiple of every k, use a large
    sentinel-free encoding via gcd = 0 -> canonical (1, 0) is LOSSY for
    zero; instead callers should use :func:`multiple_of_const` on a known
    value.  For expression analysis we keep gcd = |value| (value != 0) and
    treat 0 as (1, 0): the query API below short-circuits known literals.
    """
    if value == 0:
        return (1, 0)
    return _canon(abs(value), 0)


def _combine_add(fa: DivisibilityFact, fb: DivisibilityFact, sign: int) -> DivisibilityFact:
    g1, r1 = fa
    g2, r2 = fb
    g = _math_gcd(g1, g2)
    return _canon(g, r1 + sign * r2)


def _combine_mul(fa: DivisibilityFact, fb: DivisibilityFact) -> DivisibilityFact:
    """a*b under facts (g1,r1),(g2,r2), soundly and without inventing
    divisors.

    Writing a = k1*g1 + r1, b = k2*g2 + r2 gives
        a*b = k1*k2*g1*g2 + k1*g1*r2 + k2*g2*r1 + r1*r2.
    A modulus m is sound for the product iff it divides EVERY cross term;
    the largest such modulus dividing all three cross-term coefficients is
        m = gcd(g1*g2, g1*r2, g2*r1),
    and the residue is r1*r2 mod m.  (An earlier draft shortcut "modulus
    g1*g2 whenever either residue is 0" was WRONG: a≡0 (mod 1), b≡1 (mod 2)
    yields a*b ≡ 1 (mod 2), disproved by the exhaustive witness sweep in
    test_alignment_analysis.py — never reintroduce it.)  The dominant
    stride cases stay tight because literal right-hand sides carry residue
    0 at their own value modulus:
        name * literal c   ->  m = gcd(g1*c, g1*0, c*r1) = g1 * gcd(c, r1)
        literal c * name   ->  m = gcd(c, r1) when g1 = 1 (unknown name),
    so ``idx * 4`` with idx opaque still reports modulus 4 residue 0.
    """
    g1, r1 = fa
    g2, r2 = fb
    m = _math_gcd(g1 * g2, _math_gcd(g1 * r2, g2 * r1))
    return _canon(m, r1 * r2)


def _combine_floordiv(fa: DivisibilityFact, divisor_value: Optional[int]) -> DivisibilityFact:
    """a // b (C-truncating) divisibility: provable only for a constant
    nonzero divisor and a dividend with residue 0 at a modulus that is a
    multiple of |divisor|: then (a // d) % (g // |d|) == 0.  Everything
    else carries only the universal fact (any integer is 0 mod 1)."""
    if divisor_value is None or divisor_value == 0:
        return _UNIVERSAL
    g1, r1 = fa
    d = abs(divisor_value)
    if r1 == 0 and g1 % d == 0 and g1 > d:
        return _canon(g1 // d, 0)
    return _UNIVERSAL


def _combine_mod(fa: DivisibilityFact, divisor_value: Optional[int]) -> DivisibilityFact:
    """a % b (C remainder) for constant b == d: result's divisors derive
    from gcd(g1, |d|); residue r1 % that gcd.  Non-constant divisor:
    universal fact."""
    if divisor_value is None or divisor_value == 0:
        return _UNIVERSAL
    g1, r1 = fa
    g = _math_gcd(g1, abs(divisor_value))
    return _canon(g, r1)


# The vacuously-true fact: every integer is 0 mod 1.  Used as the stand-in
# for an unknown operand so one unknown subtree never destroys divisibility
# proven from the other (``idx * 4`` must keep the literal 4's divisors even
# when ``idx`` is opaque).
_UNIVERSAL: DivisibilityFact = (1, 0)


def _eval(node: ast.AST, env: Mapping[str, DivisibilityFact]) -> Tuple[Optional[DivisibilityFact], Optional[int]]:
    """Return (fact_or_None, known_value) for ``node``.

    ``fact`` is None when the node carries no divisibility information at
    all (unknown name, unsupported construct); binary combines substitute
    :data:`_UNIVERSAL` for it so one opaque operand never erases divisors
    proven from the other operand.  ``known_value`` is set only for integer
    literals (the analysis deliberately does not re-implement constant
    folding; that is ``static_int_analysis``'s job, and its results enter
    here through the env).
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            return None, None
        return literal_fact(node.value), node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id], None
        return None, None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        f, _ = _eval(node.operand, env)
        if isinstance(node.op, ast.UAdd) or f is None:
            return f, None
        g, r = f
        return _canon(g, -r), None
    if isinstance(node, ast.BinOp):
        fa_raw, va = _eval(node.left, env)
        fb_raw, vb = _eval(node.right, env)
        # Exact literal fold: both operands known literally -> the result is
        # an exact literal too (C-truncating div/mod, guarded against zero
        # divisors), keeping literal chains maximally tight.
        if va is not None and vb is not None:
            if isinstance(node.op, ast.Add):
                return literal_fact(va + vb), va + vb
            if isinstance(node.op, ast.Sub):
                return literal_fact(va - vb), va - vb
            if isinstance(node.op, ast.Mult):
                return literal_fact(va * vb), va * vb
            if isinstance(node.op, (ast.FloorDiv, ast.Div)) and vb != 0:
                q = abs(va) // abs(vb)
                if (va < 0) != (vb < 0):
                    q = -q
                return literal_fact(q), q
            if isinstance(node.op, ast.Mod) and vb != 0:
                qq = abs(va) // abs(vb)
                if (va < 0) != (vb < 0):
                    qq = -qq
                m = va - qq * vb
                return literal_fact(m), m
            if isinstance(node.op, ast.LShift) and 0 <= vb < 63:
                return literal_fact(va << vb), va << vb
        fa = fa_raw if fa_raw is not None else _UNIVERSAL
        fb = fb_raw if fb_raw is not None else _UNIVERSAL
        op = node.op
        # Add/Sub with an exact literal keeps the other side's full modulus:
        # (a ≡ r mod g) + 2  is  ≡ r+2 mod g, no gcd weakening.  This is
        # what makes a ragged lane offset like idx*4 + 2 provably ≡ 2 mod 4
        # (a hard no-vector answer) instead of just "even".
        if isinstance(op, (ast.Add, ast.Sub)):
            sign = +1 if isinstance(op, ast.Add) else -1
            if vb is not None:
                g, r = fa
                return _canon(g, r + sign * vb), None
            if va is not None:
                g, r = fb
                if isinstance(op, ast.Add):
                    return _canon(g, va + r), None
                return _canon(g, va - r), None
            return _combine_add(fa, fb, sign), None
        if isinstance(op, ast.Mult):
            return _combine_mul(fa, fb), None
        if isinstance(op, (ast.FloorDiv, ast.Div)):
            return _combine_floordiv(fa, vb), None
        if isinstance(op, ast.Mod):
            return _combine_mod(fa, vb), None
        if isinstance(op, ast.LShift):
            # a << b with constant b: multiply by 2**b exactly.
            if vb is not None and 0 <= vb < 63:
                g, r = fa
                return _canon(g << vb, r << vb), None
            return _UNIVERSAL, None
        # RShift, bitwise ops, pow and any other operator re-entering an
        # arithmetic slot keep only the universal fact.
        return _UNIVERSAL, None
    # Anything else (calls, attributes, subscripts, conditionals, floats):
    # no divisibility information.
    return None, None


def analyze(node: ast.AST, env: Mapping[str, DivisibilityFact]) -> Optional[DivisibilityFact]:
    """Divisibility fact of ``node`` under ``env``.

    ``env`` maps names to facts, e.g. from parameter assumptions or earlier
    proven assignments.  Returns None when the expression contains no
    analyzable information at all (a leaf unknown and no literal context);
    a combined expression with at least one informative operand returns the
    combined (possibly universal) fact instead.  Never raises on malformed
    input and never invents a fact: every reported ``(g, r)`` is a
    necessary consequence of the env facts.
    """
    try:
        fact, _ = _eval(node, env)
        return fact
    except Exception:
        # Any internal surprise must surface as "unknown", not a crash of
        # an unrelated frontend pipeline; the analysis is advisory.
        return None


def proves_multiple_of(node: ast.AST, k: int, env: Mapping[str, DivisibilityFact]) -> bool:
    """True only when the analysis proves ``node % k == 0`` (Python-mod
    class), the exact precondition query a vector-legalization check needs.

    A known literal short-circuits on its exact value (covers the value 0,
    which is a multiple of every k but has no finite gcd encoding).
    """
    if k <= 0:
        raise ValueError(f"modulus must be positive (got {k})")
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value % k == 0
    fact = analyze(node, env)
    if fact is None:
        return False
    g, r = fact
    return g % k == 0 and r % k == 0


def proves_congruent(node: ast.AST, k: int, residue: int, env: Mapping[str, DivisibilityFact]) -> bool:
    """True only when the analysis proves ``node % k == residue % k``.

    Generalizes :func:`proves_multiple_of` to non-zero residues — e.g. a
    v2 (8-byte) access packed at a fixed 8-byte phase inside a 16-byte
    tile is legal when the offset is provably ≡ 8 (mod 16) in BYTES."""
    if k <= 0:
        raise ValueError(f"modulus must be positive (got {k})")
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value % k == residue % k
    fact = analyze(node, env)
    if fact is None:
        return False
    g, r = fact
    # node % g == r  implies  node % k == residue  iff  k | g and r ≡ residue (mod k)
    return g % k == 0 and r % k == residue % k


def env_from_literal_bindings(assignments: Mapping[str, int]) -> Dict[str, DivisibilityFact]:
    """Seed an environment from known-constant names (e.g. the records a
    ``compute_static_int_consts`` run proves): each literal maps to its own
    exact fact, so its divisors propagate through products/sums."""
    return {name: literal_fact(v) for name, v in assignments.items()}
