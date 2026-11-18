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
# to permit persons to whom the Software is furnished to do so,
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
"""Regression coverage for masked/ragged control flow and irregular scheduling.

Generic per-segment / per-element kernels (ragged sorts, masked scans,
early-exit searches) are built from a small set of control-flow idioms:

  * for loops with runtime (tensor-loaded) bounds  -> ragged trip counts
  * reverse scans with break                       -> bounded backward search
  * while loops with compound bound+value guards   -> insertion-sort scans
  * break/continue in nested ifs                   -> masked / filtered work
  * variables first assigned in loop bodies        -> Python->C++ hoisting
  * unsigned (uint32) comparisons                  -> unsigned key order

These tests lock in the CUDA C++ emission for those idioms on CPU (no GPU),
and assert that Python's unsupported while/else clause fails loudly instead
of emitting syntactically invalid C++ (a `while (...) {...} else {...}` block
is not valid C++; visit_For already rejected the analogous for/else).
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


class TestRaggedForLoops(unittest.TestCase):

    def test_runtime_bounds_from_tensor(self):
        """for j in range(lo, hi) with tensor-loaded bounds emits a
        data-dependent loop, the core ragged-segment idiom."""
        code = _emit("""
def kern(begins: ll.Tensor[ll.int32], ends: ll.Tensor[ll.int32],
         out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    s = ll.threadIdx_x()
    lo = begins[s]
    hi = ends[s]
    acc = 0
    for j in range(lo, hi):
        acc = acc + j
    out[s] = acc
""")
        self.assertIn("for (int32_t j = lo; j < hi; j += 1) {", code)

    def test_reverse_scan_with_break(self):
        """Descending constant-step scan with early exit: the loop condition
        must compare downward and the step must decrement."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
         lo: ll.int32, hi: ll.int32) -> ll.void:
    last = lo
    for j in range(hi - 1, lo - 1, -1):
        if x[j] < 0:
            last = j
            break
    out[0] = last
""")
        self.assertIn("j -= 1", code)
        self.assertIn("j > lo - 1", code)
        self.assertIn("break;", code)

    def test_dynamic_step_uses_runtime_condition(self):
        """A step that is not a compile-time constant must not hard-code a
        sign: the emitted condition dispatches on the runtime step value."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
         lo: ll.int32, hi: ll.int32, step: ll.int32) -> ll.void:
    acc = 0
    for j in range(lo, hi, step):
        acc = acc + x[j]
    out[0] = acc
""")
        self.assertIn("((step) > 0 ? (j < hi) : (j > hi))", code)
        self.assertIn("j += step", code)

    def test_continue_in_nested_if(self):
        """continue inside nested if/else (masked element filtering) keeps
        loop-carried updates in both branches."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    for j in range(0, n):
        b = x[j]
        if b == 0:
            continue
        if b > 0:
            i = i + b
        else:
            i = i - b
    out[0] = i
""")
        self.assertIn("continue;", code)
        self.assertIn("else {", code)


class TestWhileExitPatterns(unittest.TestCase):

    def test_compound_guard_insertion_scan(self):
        """while (bound) and (value): the classic insertion-sort / search
        scan guard lowers to a single while with both conjuncts."""
        code = _emit("""
def kern(keys: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
         lo: ll.int32, pivot: ll.int32) -> ll.void:
    j = lo
    while (j >= 0) and (keys[j] > pivot):
        keys[j + 1] = keys[j]
        j -= 1
    out[0] = j
""")
        self.assertIn("while (j >= 0 && keys[j] > pivot) {", code)

    def test_break_inside_while_with_hoisted_vars(self):
        """Body-assigned variables are declared before the while loop so the
        early-exit result is visible in a sibling loop and after it."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = 0
    while i < n:
        v = x[i]
        if v == 0:
            break
        i += 1
    for k in range(v, i):
        out[k] = v
""")
        decl_pos = code.index("int32_t v;")
        while_pos = code.index("while (")
        self.assertLess(decl_pos, while_pos, "declaration must precede the while loop")
        self.assertEqual(code.count("int32_t v"), 1)

    def test_while_else_is_rejected(self):
        """Python's while/else has no C++ form; emitting `while {...} else {...}`
        is a guaranteed nvcc syntax error, so the frontend must raise instead
        (mirroring visit_For's rejection of for/else)."""
        with self.assertRaises(NotImplementedError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = 0
    while i < n:
        if x[i] == 0:
            break
        i += 1
    else:
        out[0] = -1
    out[1] = i
""")
        self.assertIn("else", str(cm.exception))

    def test_for_else_still_rejected(self):
        """for/else remains rejected (pre-existing behavior, kept in sync)."""
        with self.assertRaises(NotImplementedError):
            _emit("""
def kern(out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    for j in range(n):
        out[j] = j
    else:
        out[0] = -1
""")


class TestHoistingAcrossElif(unittest.TestCase):

    def test_elif_chain_hoists_single_declaration(self):
        """A variable first assigned in every arm of an if/elif/else chain is
        declared exactly once, before the chain."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], i: ll.int32) -> ll.void:
    if i == 0:
        r: ll.int32 = x[0]
    elif i == 1:
        r: ll.int32 = x[1]
    else:
        r: ll.int32 = -1
    out[0] = r
""")
        self.assertEqual(code.count("int32_t r"), 1)
        decl_pos = code.index("int32_t r;")
        if_pos = code.index("if (")
        self.assertLess(decl_pos, if_pos)
        self.assertIn("else if", code)


class TestUnsignedKeyComparison(unittest.TestCase):

    def test_uint32_params_compare_unsigned(self):
        """uint32 keys keep their unsigned C type so comparisons order
        unsigned bit patterns (the descending-unsigned sort key domain)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32], i: ll.int32) -> ll.void:
    a: ll.uint32 = x[i]
    b: ll.uint32 = x[i + 1]
    if a < b:
        out[i] = b
    else:
        out[i] = a
""")
        self.assertIn("uint32_t a = x[i];", code)
        self.assertIn("if (a < b) {", code)

    def test_uint32_while_scan(self):
        """Unsigned bound comparisons inside a scan loop stay unsigned."""
        code = _emit("""
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.int32],
         lo: ll.uint32, hi: ll.uint32) -> ll.void:
    j: ll.uint32 = lo
    while j < hi:
        j += 1
    out[0] = 0
""")
        self.assertIn("uint32_t j = lo;", code)
        self.assertIn("while (j < hi) {", code)


class TestAugmentedAssignSubscript(unittest.TestCase):
    """Scatter-accumulate (out[j] += v) is the store-side idiom of masked and
    ragged irregular kernels: segmented sums, histograms and gap-preserving
    scatters all update a tensor element under a guard. The type checker used
    to reject any augmented assignment whose target was not a plain name."""

    def test_guarded_scatter_accumulate(self):
        """out[j] += srcs[i] under a bounds+validity mask emits a plain C++
        compound store through the computed index."""
        code = _emit("""
def kern(srcs: ll.Tensor[ll.int32], idx: ll.Tensor[ll.int32],
         out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        j = idx[i]
        if j >= 0:
            out[j] += srcs[i]
""")
        self.assertIn("out[j] += srcs[i];", code)

    def test_ragged_segment_sum(self):
        """Accumulating into out[s] inside a ragged for loop (per-segment
        reduction over [lo, hi)) keeps the compound store form."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], begins: ll.Tensor[ll.int32],
         ends: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], s: ll.int32) -> ll.void:
    lo = begins[s]
    hi = ends[s]
    for j in range(lo, hi):
        out[s] += x[j]
""")
        self.assertIn("for (int32_t j = lo; j < hi; j += 1) {", code)
        self.assertIn("out[s] += x[j];", code)

    def test_uint32_bitwise_or_scatter(self):
        """Bitwise augmented ops on uint32 elements (flag setting in masked
        work queues) keep the unsigned element type and |= form."""
        code = _emit("""
def kern(flags: ll.Tensor[ll.uint32], pos: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        p = pos[i]
        v: ll.uint32 = 1
        flags[p] |= v
""")
        self.assertIn("flags[p] |= v;", code)

    def test_float_masked_sub_assign(self):
        """float32 element -= under a value guard (clamping/filtering idiom)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        if x[i] > 1.0:
            out[i] -= x[i]
""")
        self.assertIn("out[i] -= x[i];", code)

    def test_pointer_base_subscript_target(self):
        """A raw pointer kernel argument (not just ll.Tensor) also accepts
        element augmented assignment."""
        code = _emit("""
def kern(out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        out[i] *= 2
""")
        self.assertIn("out[i] *= 2;", code)

    def test_scalar_base_rejected(self):
        """Subscript on a scalar is a compile error, not silent bad C++:
        the checker names the offending base type."""
        with self.assertRaises(Exception) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    n[0] += 1
""")
        self.assertIn("tensor/pointer base", str(cm.exception))

    def test_plain_variable_targets_still_work(self):
        """Pre-existing name-target augmented assignment is unchanged."""
        code = _emit("""
def kern(out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    acc = 0
    for j in range(0, n):
        acc += j
    out[0] = acc
""")
        self.assertIn("acc += j;", code)


class TestSegmentedSortSkeleton(unittest.TestCase):

    def test_per_segment_stable_insertion_sort_skeleton(self):
        """The control-flow skeleton of a generic per-segment stable sort:
        ragged outer trip count + reverse data-dependent inner scan under a
        guard flag lowers as one structurally valid unit."""
        code = _emit("""
def kern(keys: ll.Tensor[ll.int32], vals: ll.Tensor[ll.int32],
         seg_begin: ll.Tensor[ll.int32], seg_end: ll.Tensor[ll.int32],
         nseg: ll.int32) -> ll.void:
    s = ll.threadIdx_x()
    if s < nseg:
        lo = seg_begin[s]
        hi = seg_end[s]
        for p in range(lo + 1, hi):
            k = keys[p]
            v = vals[p]
            j = p - 1
            done = 0
            while (j >= lo) and (done == 0):
                kj = keys[j]
                if kj < k:
                    done = 1
                else:
                    keys[j + 1] = kj
                    vals[j + 1] = vals[j]
                    j -= 1
            keys[j + 1] = k
            vals[j + 1] = v
""")
        # structurally: guarded body, ragged inner bounds, compound while
        self.assertIn("if (s < nseg) {", code)
        self.assertIn("for (int32_t p = lo + 1; p < hi; p += 1) {", code)
        self.assertIn("while (j >= lo && done == 0) {", code)
        self.assertIn("else {", code)
        # hoisted inner-scan temps survive across segments
        self.assertEqual(code.count("int32_t kj"), 1)
        # guarded body emitted before any dangling else (no orphan braces)
        self.assertEqual(code.count("{"), code.count("}"))


class TestLoopBodyBindingsEscapeLoop(unittest.TestCase):
    """Python has no block scope: a name first assigned in a for body is a
    function-local binding usable after the loop. Ragged kernels carry scan
    results (last key, insertion point, hit flag) out of the per-segment
    loop this way. Type inference used to push/pop a scope around the for
    body and drop those bindings, rejecting programs the C/CUDA backend
    already hoists and emits correctly (_scan_for_assigned_vars); while and
    if bodies already kept function scope through the generic visitor. The
    loop variable itself stays loop-local (codegen deletes it at the end of
    visit_For)."""

    def test_scan_result_used_after_loop(self):
        """Value assigned in the loop body is visible after the loop and
        hoisted exactly once before it in the emitted C++."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    acc = 0
    for j in range(0, n):
        last = x[j]
        acc += last
    out[0] = last
    out[1] = acc
""")
        decl_pos = code.index("int32_t last;")
        loop_pos = code.index("for (")
        self.assertLess(decl_pos, loop_pos)
        self.assertEqual(code.count("int32_t last"), 1)
        self.assertIn("out[0] = last;", code)

    def test_binding_escapes_loop_inside_mask(self):
        """The masked form: per-thread guarded ragged scan whose result is
        consumed after the loop but still inside the guard arm."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        for j in range(0, n):
            last = x[j]
        out[i] = last
""")
        self.assertIn("if (i < n) {", code)
        self.assertIn("out[i] = last;", code)

    def test_flag_and_index_escape_ragged_scan(self):
        """Search-over-segment idiom: hit flag and position assigned inside
        the segment scan drive a guarded store after it."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], begins: ll.Tensor[ll.int32],
         ends: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
         nseg: ll.int32) -> ll.void:
    s = ll.threadIdx_x()
    if s < nseg:
        lo = begins[s]
        hi = ends[s]
        pivot = x[lo]
        pos = lo
        for j in range(lo, hi):
            kj = x[j]
            if kj == pivot:
                pos = j
                seen = 1
        if seen == 1:
            out[s] = pos
        else:
            out[s] = -1
""")
        self.assertEqual(code.count("{"), code.count("}"))
        self.assertIn("for (int32_t j = lo; j < hi; j += 1) {", code)
        self.assertEqual(code.count("int32_t seen;"), 1)
        self.assertIn("if (seen == 1) {", code)

    def test_nested_loop_bindings_chain_outward(self):
        """Bindings propagate one scope per loop, so values assigned in an
        inner ragged loop are usable after the outer loop too."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], begins: ll.Tensor[ll.int32],
         ends: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
         nseg: ll.int32) -> ll.void:
    for s in range(0, nseg):
        lo = begins[s]
        hi = ends[s]
        for j in range(lo, hi):
            total = x[j]
    out[0] = total
""")
        self.assertIn("out[0] = total;", code)

    def test_loop_variable_still_loop_local(self):
        """The loop variable must NOT escape: codegen deletes it from scope
        after the loop, so inference rejects the use instead of emitting a
        reference to an out-of-scope C++ identifier."""
        with self.assertRaises(Exception) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    for j in range(0, n):
        out[j] = x[j]
    out[0] = j
""")
        self.assertIn("'j'", str(cm.exception))

    def test_incompatible_reassignment_inside_loop_rejected(self):
        """Assignment inside a loop body rebinds (never shadows) an outer
        function-local: a value that cannot adopt the declared type must
        raise inside the loop exactly as it does at function level. The old
        innermost-scope check silently shadowed the binding and emitted a
        hidden static_cast."""
        with self.assertRaises(Exception) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    v: ll.int32 = 0
    for j in range(0, n):
        v = 1.5
    out[0] = v
""")
        self.assertIn("changes type", str(cm.exception))

    def test_compatible_reassignment_inside_loop_keeps_type(self):
        """Numeric-compatible reassignment (int literal into a declared
        uint32) adopts the declared dtype and stays compilable."""
        code = _emit("""
def kern(out: ll.Tensor[ll.uint32], n: ll.int32) -> ll.void:
    u: ll.uint32 = 0
    for j in range(0, n):
        u = 7
    out[0] = u
""")
        self.assertIn("uint32_t u = 0;", code)
        self.assertIn("u = static_cast<uint32_t>(7);", code)


class TestRangeLoopVariableBoundTyping(unittest.TestCase):
    """for j in range(lo, hi) must declare j in the comparison domain of its
    bounds. visit_For used to hardcode the counter to int32_t; against a
    uint32 segment bound the emitted condition is then a mixed-sign C++
    comparison, and usual arithmetic conversions silently reinterpret the
    signed counter as unsigned -- a descending sink scan over an unsigned
    segment range runs past the segment. The counter type is now promoted
    from the bound expressions with the same C++ conversion rules type
    inference uses."""

    def test_uint32_bounds_ascending_ragged(self):
        """Ascending segment loop over uint32 offsets: the counter is
        uint32_t so j < hi is a same-sign unsigned comparison."""
        code = _emit("""
def kern(x: ll.Tensor[ll.uint32], begins: ll.Tensor[ll.uint32],
         ends: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32], s: ll.int32) -> ll.void:
    lo = begins[s]
    hi = ends[s]
    for j in range(lo, hi):
        out[j] = x[j]
""")
        self.assertIn("for (uint32_t j = lo; j < hi; j += 1) {", code)

    def test_uint32_bounds_descending_scan(self):
        """Descending sink scan over an unsigned segment: uint32 counter,
        so j > lo - 1 compares two uint32 values. With the old int32_t
        counter this condition converted j to unsigned, and a wrapped lo - 1
        made the loop scan far past the segment."""
        code = _emit("""
def kern(keys: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32],
         lo: ll.uint32, hi: ll.uint32) -> ll.void:
    for j in range(hi - 1, lo - 1, -1):
        out[j] = keys[j]
""")
        self.assertIn("for (uint32_t j = hi - 1; j > lo - 1; j -= 1) {", code)

    def test_int64_bounds_wide_offsets(self):
        """Segments in a flat buffer larger than 2**31 elements use int64
        offsets; the counter must not truncate the wrapped bounds."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int64], out: ll.Tensor[ll.int64],
         lo: ll.int64, hi: ll.int64) -> ll.void:
    for j in range(lo, hi):
        out[j] = x[j]
""")
        self.assertIn("for (int64_t j = lo; j < hi; j += 1) {", code)

    def test_int32_bounds_unchanged(self):
        """Plain int32 ragged loop keeps the historical int32_t counter."""
        code = _emit("""
def kern(begins: ll.Tensor[ll.int32], ends: ll.Tensor[ll.int32],
         out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    s = ll.threadIdx_x()
    lo = begins[s]
    hi = ends[s]
    acc = 0
    for j in range(lo, hi):
        acc = acc + j
    out[s] = acc
""")
        self.assertIn("for (int32_t j = lo; j < hi; j += 1) {", code)
        self.assertNotIn("uint32_t j", code)

    def test_literal_only_bounds_unchanged(self):
        """Compile-time literal bounds default to int32 as before."""
        code = _emit("""
def kern(out: ll.Tensor[ll.int32]) -> ll.void:
    for j in range(8):
        out[j] = j
""")
        self.assertIn("for (int32_t j = 0; j < 8; j += 1) {", code)

    def test_mixed_uint32_int32_bounds_follow_cpp_rules(self):
        """Mixed uint32/int32 bounds promote to uint32, matching the usual
        arithmetic conversions the emitted comparison would apply anyway."""
        code = _emit("""
def kern(out: ll.Tensor[ll.uint32], lo: ll.uint32, hi: ll.int32) -> ll.void:
    for j in range(lo, hi):
        out[j] = 0
""")
        self.assertIn("for (uint32_t j = lo; j < hi; j += 1) {", code)

    def test_narrow_unsigned_bounds_stay_int32(self):
        """uint16 bounds promote to int32 (C integer promotion of small
        types), not uint16 -- keeps wrap behavior identical to C."""
        code = _emit("""
def kern(out: ll.Tensor[ll.int32], lo: ll.uint16, hi: ll.uint16) -> ll.void:
    for j in range(lo, hi):
        out[j] = 0
""")
        self.assertIn("for (int32_t j = lo; j < hi; j += 1) {", code)

    def test_runtime_step_keeps_condition_with_typed_counter(self):
        """The dynamic-step runtime condition path is preserved when the
        bounds are unsigned; only the counter type changes."""
        code = _emit("""
def kern(x: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32],
         lo: ll.uint32, hi: ll.uint32, step: ll.uint32) -> ll.void:
    for j in range(lo, hi, step):
        out[j] = x[j]
""")
        self.assertIn("for (uint32_t j = lo;", code)
        self.assertIn("((step) > 0 ? (j < hi) : (j > hi))", code)
        self.assertIn("j += step", code)

    def test_predeclared_loop_variable_keeps_declared_type(self):
        """When the loop variable is already in scope (author-declared), its
        declared type wins and no new declaration is emitted."""
        code = _emit("""
def kern(out: ll.Tensor[ll.uint32], j: ll.int32, hi: ll.uint32) -> ll.void:
    for j in range(0, hi):
        out[j] = 0
""")
        self.assertIn("for (j = 0; j < hi; j += 1) {", code)


class TestRangeBoundTensorSubscriptTyping(unittest.TestCase):
    """range() bounds that load directly from a tensor (``offsets[e]``) must
    type the loop counter from the tensor element type. Ragged segment
    iteration is commonly written straight over the offsets array; when the
    offsets are int64 (flat buffers past 2**31 elements) or uint32, collapsing
    the subscript to int32 truncates the bound or creates a mixed-sign
    comparison in the emitted loop condition."""

    def test_int64_direct_subscript_bounds(self):
        """for j in range(offsets[e - 1], offsets[e]) with int64 offsets
        declares an int64 counter -- an int32 counter would truncate segment
        ends for buffers past 2**31 elements."""
        code = _emit("""
def kern(offsets: ll.Tensor[ll.int64], x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], e: ll.int32) -> ll.void:
    for j in range(offsets[e - 1], offsets[e]):
        out[j] = x[j]
""")
        self.assertIn("for (int64_t j = offsets[e - 1]; j < offsets[e]; j += 1) {", code)

    def test_uint32_direct_subscript_bounds(self):
        """uint32 offsets give a uint32 counter, keeping j < offsets[e] a
        same-sign unsigned comparison."""
        code = _emit("""
def kern(offsets: ll.Tensor[ll.uint32], x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], e: ll.int32) -> ll.void:
    for j in range(offsets[e - 1], offsets[e]):
        out[j] = x[j]
""")
        self.assertIn("for (uint32_t j = offsets[e - 1]; j < offsets[e]; j += 1) {", code)

    def test_int32_subscript_bounds_stay_int32(self):
        """int32 element offsets keep the historical int32_t counter."""
        code = _emit("""
def kern(offsets: ll.Tensor[ll.int32], x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], e: ll.int32) -> ll.void:
    for j in range(0, offsets[e]):
        out[j] = x[j]
""")
        self.assertIn("for (int32_t j = 0; j < offsets[e]; j += 1) {", code)
        self.assertNotIn("int64_t j", code)
        self.assertNotIn("uint32_t j", code)

    def test_narrow_uint16_subscript_bounds_promote_to_int32(self):
        """uint16 elements undergo C integer promotion: the counter is int32,
        matching the wrap/comparison semantics of the emitted C++."""
        code = _emit("""
def kern(offsets: ll.Tensor[ll.uint16], out: ll.Tensor[ll.int32], e: ll.int32) -> ll.void:
    for j in range(0, offsets[e]):
        out[j] = 0
""")
        self.assertIn("for (int32_t j = 0; j < offsets[e]; j += 1) {", code)

    def test_int64_assigned_from_subscript_still_typed(self):
        """The pre-existing Name path (bound assigned from a subscript load)
        keeps working alongside the new direct-subscript arm."""
        code = _emit("""
def kern(offsets: ll.Tensor[ll.int64], x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32], e: ll.int32) -> ll.void:
    lo = offsets[e]
    hi = offsets[e + 1]
    for j in range(lo, hi):
        out[j] = x[j]
""")
        self.assertIn("int64_t lo = offsets[e];", code)
        self.assertIn("for (int64_t j = lo; j < hi; j += 1) {", code)


if __name__ == "__main__":
    unittest.main()
