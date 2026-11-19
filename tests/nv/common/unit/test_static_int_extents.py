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
"""Compile-time-constant static/shared array extents and the implicit
mutation-through-intrinsic-output rule.

CUDA requires ``__shared__`` / local C array extents to be integer constant
expressions.  Frontends pin workload bounds as typed integer locals
initialized from literals/arithmetic (``MAX_N: ll.int32 = 52``) and pass them
to ``alloc_shared_memory``.  Emitting those as plain ``int32_t`` locals makes
the array a variable-length array with static storage duration, which nvcc
rejects.  The provably-static analysis
(``codegen/visitors/static_int_analysis.py``) emits a local that is constant,
singly-defined, never reassigned, never escape-through-a-call, and in range
as ``static constexpr`` so the extent is a legal constant expression; nvcc
re-verifies this, so a wrong conclusion is a hard error, never silent
miscompilation.

Two honest-failure guarantees are locked here:

* a runtime-dependent / reassigned / out-of-range extent is NOT frozen into a
  guessed constant — it raises a source-located ``ValueError`` before any
  invalid CUDA reaches nvcc;
* a typed local mutated only *implicitly* through an intrinsic output
  pointer (``gmem_ld_v4s32`` emits ``&r0`` and stores through it) is never
  marked ``constexpr``, even though the plain AST Store scan sees a single
  assignment — otherwise nvcc rejects taking the address.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    return codegen_cuda(ast.parse(source), ctx={"ll": ll}, emit_header=False)


class TestStaticIntExtentsPositive(unittest.TestCase):
    """Provably-static integer extents become legal constant expressions."""

    def test_literal_and_name_extents(self):
        code = _emit("""
def kern(q: ll.Tensor[ll.bfloat16], lengths: ll.Tensor[ll.int32]) -> ll.void:
    MAX_N: ll.int32 = 52
    ll.alloc_shared_memory("off_tbl", ll.int32, MAX_N)
    ll.alloc_shared_memory("s_rank", ll.int32, 192)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < MAX_N:
        off_tbl[tid] = tid
    ll.__syncthreads()
""")
        self.assertIn("static constexpr int32_t MAX_N = 52;", code)
        self.assertIn("__shared__ int32_t off_tbl[MAX_N];", code)
        # A bare literal extent is already a constant expression.
        self.assertIn("__shared__ int32_t s_rank[192];", code)

    def test_arithmetic_extent_over_statics(self):
        code = _emit("""
def kern(q: ll.Tensor[ll.bfloat16]) -> ll.void:
    HEAD_DIM: ll.int32 = 128
    TILE: ll.int32 = 8
    STAGES: ll.int32 = 2
    ll.alloc_shared_memory("tile_sm", ll.bfloat16, STAGES * TILE * HEAD_DIM)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < TILE:
        tile_sm[tid] = q[tid]
    ll.__syncthreads()
""")
        self.assertIn("static constexpr int32_t HEAD_DIM = 128;", code)
        self.assertIn("static constexpr int32_t TILE = 8;", code)
        self.assertIn("static constexpr int32_t STAGES = 2;", code)
        # Extent arithmetic over statics is emitted verbatim and is an ICE.
        self.assertIn("__shared__ __nv_bfloat16 tile_sm[STAGES * TILE * HEAD_DIM];", code)

    def test_static_bound_shared_by_range_and_alloc(self):
        """One typed static bound serves both a ``range`` loop bound and the
        static shared allocation.  The real builtin ``range`` reads its bound
        only; it must not cancel the constant proof, so the extent stays a
        legal constant expression and nvcc accepts the array."""
        code = _emit("""
def kern(q: ll.Tensor[ll.bfloat16]) -> ll.void:
    STAGES: ll.int32 = 2
    TILE: ll.int32 = 8
    HEAD_DIM: ll.int32 = 128
    ll.alloc_shared_memory("tile_sm", ll.bfloat16, STAGES * TILE * HEAD_DIM)
    tid: ll.int32 = ll.threadIdx_x()
    for s in range(STAGES):
        tile_sm[tid] = q[tid]
    ll.__syncthreads()
""")
        self.assertIn("static constexpr int32_t STAGES = 2;", code)
        self.assertIn("static constexpr int32_t TILE = 8;", code)
        self.assertIn("static constexpr int32_t HEAD_DIM = 128;", code)
        self.assertIn("__shared__ __nv_bfloat16 tile_sm[STAGES * TILE * HEAD_DIM];", code)
        # The loop bound is the same constant, not a captured mutable local.
        self.assertIn("for (int32_t s = 0; s < STAGES; s += 1)", code)

    def test_div_mod_bitwise_fold(self):
        """Constant folding uses C truncation for // and %, and supports the
        documented bitwise/shift operators."""
        source = """
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    A: ll.int32 = 130
    HALF: ll.int32 = A // 2
    REM: ll.int32 = A % 32
    B: ll.int32 = (1 << 6) | (A & 31)
    ll.alloc_shared_memory("s", ll.int32, HALF + REM)
    ll.alloc_shared_memory("t", ll.int32, B)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < HALF:
        s[tid] = q[tid]
    ll.__syncthreads()
"""
        code = _emit(source)
        # Assert the analyzer's integer semantics, independently of C++
        # whitespace or optional parentheses. Python // emits C++ / here.
        from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter
        emitter = CudaEmitter(ctx={"ll": ll}, all_types={})
        emitter._begin_function_static_consts(ast.parse(source).body[0])
        try:
            values = {name: record.value for name, record in emitter._static_int_consts.items()}
            self.assertEqual({name: values[name]
                              for name in ("A", "HALF", "REM", "B")}, {"A": 130, "HALF": 65, "REM": 2, "B": 66})
        finally:
            emitter._end_function_static_consts()
        for name in ("A", "HALF", "REM", "B"):
            self.assertRegex(code, rf"\bstatic\s+constexpr\s+int32_t\s+{name}\s*=")
        self.assertRegex(code, r"__shared__\s+int32_t\s+s\s*\[\s*HALF\s*\+\s*REM\s*\]\s*;")
        self.assertRegex(code, r"__shared__\s+int32_t\s+t\s*\[\s*B\s*\]\s*;")


class TestIntrinsicOutputMutationNotConstexpr(unittest.TestCase):
    """gmem_ld_v4s32 writes r0..r3 via &rN; they must stay plain locals."""

    def _src(self) -> str:
        return """
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    r0: ll.int32 = 0
    r1: ll.int32 = 0
    r2: ll.int32 = 0
    r3: ll.int32 = 0
    tid: ll.int32 = ll.threadIdx_x()
    ll.gmem_ld_v4s32(q, tid * 4, r0, r1, r2, r3)
    ll.gmem_st_v4s32(q, tid * 4, r0, r1, r2, r3)
"""

    def test_output_lanes_emit_plain_locals(self):
        code = _emit(self._src())
        for lane in ("r0", "r1", "r2", "r3"):
            self.assertIn(f"int32_t {lane} = 0;", code)
            self.assertNotIn(f"static constexpr int32_t {lane}", code)

    def test_output_lanes_not_marked_static(self):
        # Whole-source safety: no output-pointer lane is ever constexpr.
        code = _emit(self._src())
        self.assertNotIn("static constexpr", code)
        # ...but the address-taken call is still emitted, so the store is real.
        self.assertIn("gmem_ld_v4s32_fn(q + tid * 4, &r0, &r1, &r2, &r3);", code)

    def test_store_value_lanes_default_conservative(self):
        # gmem_st_v4s32 declares NO memory effects, so its value lanes carry
        # no read-only proof: with the semantic repair, rendered C++ text is
        # never purity evidence, and the lanes default to WRITTEN/unknown —
        # plain locals, valid C++, runtime values unchanged.  The dedicated
        # instruction intrinsic remains semantically a by-value store; what
        # changed is only that no STATIC-CONST proof is derived from its
        # template text.  (The deliberately asymmetric positive: identity-
        # proven consumers such as ``range``/alloc extents do not touch K.)
        code = _emit("""
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    K: ll.int32 = 7
    r0: ll.int32 = 1
    r1: ll.int32 = 2
    r2: ll.int32 = 3
    r3: ll.int32 = 4
    tid: ll.int32 = ll.threadIdx_x()
    ll.gmem_st_v4s32(q, tid * 4, r0, r1, r2, r3)
    if tid < K:
        q[tid] = K
""")
        self.assertIn("static constexpr int32_t K = 7;", code)
        for lane, val in (("r0", 1), ("r1", 2), ("r2", 3), ("r3", 4)):
            self.assertIn(f"int32_t {lane} = {val};", code)
            self.assertNotIn(f"constexpr int32_t {lane}", code)


class TestIntrinsicReferenceAndAssignOutputs(unittest.TestCase):
    """Two more public output forms must keep single-initialized outputs
    writable (plain locals), never ``static constexpr``:

    * ``gmem_ld_i32x4`` emits a helper whose C++ signature takes the outputs
      by *mutable reference* (``int& r0``) and assigns ``r0 = value.x;``,
      while the call site passes them plain (``{r0}``) — no ``&`` appears;
    * the 3-arg ``gmem_ld_bf16x2_u32`` emits a whole-statement assignment
      ``(bits) = <load>`` with no address syntax at all.
    """

    def test_i32x4_reference_outputs_plain_locals(self):
        code = _emit("""
def kern(q: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32]) -> ll.void:
    r0: ll.int32 = 0
    r1: ll.int32 = 0
    r2: ll.int32 = 0
    r3: ll.int32 = 0
    tid: ll.int32 = ll.threadIdx_x()
    ll.gmem_ld_i32x4(q, tid * 4, r0, r1, r2, r3)
    ll.gmem_st_i32x4(out, tid * 4, r0, r1, r2, r3)
""")
        for lane in ("r0", "r1", "r2", "r3"):
            self.assertIn(f"int32_t {lane} = 0;", code)
        self.assertNotIn("static constexpr", code)
        # The reference-taking helper call is still emitted (the write lands).
        self.assertIn("gmem_ld_i32x4_fn(q + tid * 4, r0, r1, r2, r3);", code)

    def test_bf16x2_u32_assign_output_plain_local(self):
        code = _emit("""
def kern(q: ll.Tensor[ll.bfloat16], out: ll.Tensor[ll.uint32]) -> ll.void:
    bits: ll.uint32 = 0
    tid: ll.int32 = ll.threadIdx_x()
    ll.gmem_ld_bf16x2_u32(q, tid, bits)
    out[tid] = bits
""")
        self.assertIn("uint32_t bits = 0;", code)
        self.assertNotIn("static constexpr", code)
        # The direct-assignment load is emitted; the store of `bits` is real.
        self.assertIn("(bits) = *reinterpret_cast<const uint32_t*>((q) + 2 * (tid));", code)


class TestCalleeBindingUnification(unittest.TestCase):
    """Read-only builtin recognition keys on callee IDENTITY plus binding
    state, never on spelling: the resolved real builtin object qualifies,
    the absent-from-context name falls back to the source language's
    builtin scope, and every present-but-different / shadowed binding is
    conservative (diagnosed non-constant extent, never a frozen guess)."""

    _SRC = """
def kern(q: ll.Tensor[ll.bfloat16]) -> ll.void:
    BOUND: ll.int32 = 8
    ll.alloc_shared_memory("tile", ll.bfloat16, BOUND)
    tid: ll.int32 = ll.threadIdx_x()
    for s in range(BOUND):
        tile[tid] = q[tid]
    ll.__syncthreads()
"""

    def _emit_with(self, extra_ctx) -> str:
        ctx = {"ll": ll}
        ctx.update(extra_ctx)
        return codegen_cuda(ast.parse(self._SRC), ctx=ctx, emit_header=False)

    def test_absent_name_resolves_real_builtin(self):
        # The `range` loop form bound by nothing in ctx refers to the real
        # builtin: the loop bound is read by value and stays provably static.
        code = self._emit_with({})
        self.assertIn("static constexpr int32_t BOUND = 8;", code)
        self.assertIn("__shared__ __nv_bfloat16 tile[BOUND];", code)

    def test_explicit_real_builtin_binding(self):
        # ctx['range'] = builtins.range IS the identical real object (an
        # explicitly bound actual builtin): signature inspection of the
        # builtin type must not disqualify the read-only bound.
        import builtins
        code = self._emit_with({"range": builtins.range})
        self.assertIn("static constexpr int32_t BOUND = 8;", code)
        self.assertIn("__shared__ __nv_bfloat16 tile[BOUND];", code)

    def test_present_none_binding_is_not_absent(self):
        # ctx['range'] = None is a PRESENT unknown binding, not an absent
        # name: no read-only fallback, extent is honestly diagnosed.
        with self.assertRaises(ValueError) as cm:
            self._emit_with({"range": None})
        self.assertIn("non-constant element count", str(cm.exception))

    def test_custom_object_binding_conservative(self):
        # A user callable merely NAMED range (not the builtin object) never
        # receives read-only treatment by spelling.
        with self.assertRaises(ValueError) as cm:
            self._emit_with({"range": lambda *a: []})
        self.assertIn("non-constant element count", str(cm.exception))

    def test_unresolvable_ast_binding_conservative(self):
        # A ctx entry the resolver leaves as an AST fragment is an unknown
        # callee, not the builtin of the same spelling.
        with self.assertRaises(ValueError) as cm:
            self._emit_with({"range": ast.parse("x").body[0]})
        self.assertIn("non-constant element count", str(cm.exception))

    def test_equality_spoofing_object_never_impersonates(self):
        # A custom ctx object whose __eq__ claims equality with the real
        # builtin must NOT receive read-only treatment: membership is by
        # object identity, so the spoofer stays an unknown callee and the
        # extent is honestly diagnosed.
        import builtins

        class RangeTwin:

            def __eq__(self, other):
                return other is builtins.range

            def __hash__(self):
                return hash(builtins.range)

            def __call__(self, *a):
                return []

        with self.assertRaises(ValueError) as cm:
            self._emit_with({"range": RangeTwin()})
        self.assertIn("non-constant element count", str(cm.exception))
        self.assertTrue(RangeTwin() == builtins.range)  # sanity: spoof real

    def _statics(self, source, extra_ctx):
        from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter
        ctx = {"ll": ll}
        ctx.update(extra_ctx)
        emitter = CudaEmitter(ctx=ctx, all_types={})
        func = ast.parse(source).body[0]
        emitter._begin_function_static_consts(func)
        try:
            return set(emitter._static_int_consts)
        finally:
            emitter._end_function_static_consts()

    def test_parameter_shadowing_range_conservative(self):
        # A function PARAMETER named `range` shadows the builtin in lexical
        # scope even when ctx itself binds the real object.
        import builtins
        statics = self._statics(
            """
def kern(q: ll.Tensor[ll.bfloat16], range: ll.int32) -> ll.void:
    BOUND: ll.int32 = 8
    ll.alloc_shared_memory("tile", ll.bfloat16, BOUND)
    for s in range(BOUND):
        pass
""", {"range": builtins.range})
        self.assertNotIn("BOUND", statics)

    def test_local_binding_shadowing_range_conservative(self):
        # A local assignment to `range` anywhere in the body is a lexical
        # rebinding: the loop call target is not provably the builtin.
        import builtins
        statics = self._statics(
            """
def kern(q: ll.Tensor[ll.bfloat16]) -> ll.void:
    range = ll.gmem_st_v4s32
    BOUND: ll.int32 = 8
    ll.alloc_shared_memory("tile", ll.bfloat16, BOUND)
    for s in range(BOUND):
        pass
""", {"range": builtins.range})
        self.assertNotIn("BOUND", statics)


class TestUnknownEffectsStayConservative(unittest.TestCase):
    """Rendered C++ text is never purity evidence.  Read-only needs an
    explicit intrinsic/IR R-effect or genuine primitive semantics (the
    identity-checked ``range`` builtin); every parameter with no declared
    effect — successfully rendered, asm-free, name spelled in the output,
    no write-regex match — is unknown-effect and keeps its whole-Name
    argument mutable/conservative."""

    def _statics(self, source, extra_ctx):
        from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter
        ctx = {"ll": ll}
        ctx.update(extra_ctx)
        emitter = CudaEmitter(ctx=ctx, all_types={})
        func = ast.parse(source).body[0]
        emitter._begin_function_static_consts(func)
        try:
            return set(emitter._static_int_consts)
        finally:
            emitter._end_function_static_consts()

    def test_inline_asm_template_never_proved_pure(self):
        # `asm volatile` may store through PTX-only spellings; with no
        # declared effect the value argument is unknown-effect and escapes.
        from little_kernel.language.builtin_base import builtin, Builtin

        def cg(mem, index, v):
            body = ('__device__ void sneaky_fn(int* ptr, int v) {'
                    ' asm volatile("st.global.b32 [%0], %1;"'
                    ' :: "l"(ptr), "r"(0) : "memory"); }')
            return Builtin(body=body, includes=[], return_val="sneaky_fn({mem}, {v})")

        @builtin(eval_return_type=ll.void, codegen_func=cg)
        def sneaky(mem, index, v):
            raise RuntimeError("should not be called in compilation")

        statics = self._statics(
            """
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    V: ll.int32 = 7
    tid: ll.int32 = ll.threadIdx_x()
    sneaky(q, tid, V)
""", {"sneaky": sneaky})
        self.assertNotIn("V", statics)

    def test_unknown_helper_reference_callback_stays_conservative(self):
        # The concrete counterexample: a successfully rendered, asm-free
        # callback whose C++ helper mutates the caller through a RENAMED
        # mutable reference (`renamed`, not `out`).  No `&out`, no `out =`,
        # no out-increment appears; the word `out` IS present as a plain
        # call argument.  None of that textual evidence proves pass-by-value,
        # so `out` keeps its mutable local emission (never `constexpr`,
        # which would make the implicit `int&` binding ill-formed).
        from little_kernel.language.builtin_base import builtin, Builtin

        def cg(out):
            body = ("__device__ void bump(int& renamed) { ++renamed; }")
            return Builtin(body=body, includes=[], return_val="bump({out})")

        @builtin(eval_return_type=ll.void, codegen_func=cg)
        def bump_cb(out):
            raise RuntimeError("should not be called in compilation")

        code = codegen_cuda(
            ast.parse("""
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    out: ll.int32 = 7
    bump_cb(out)
    q[out] = out
"""), ctx={"ll": ll, "bump_cb": bump_cb}, emit_header=False)
        # Renders successfully, asm-free, name spelled in the output — and
        # NONE of that is purity evidence: `out` stays a mutable local so
        # the helper's `int& renamed` binding is well-formed.
        self.assertIn("bump", code)
        self.assertIn("int32_t out = 7;", code)
        self.assertNotIn("constexpr int32_t out", code)

    def test_declared_write_effect_marks_operand(self):
        # An explicit W/RW memory effect on an operand marks that parameter
        # even when the rendered template shows no C++ write spelling.
        from little_kernel.language.builtin_base import builtin, Builtin
        from little_kernel.core.memory_effects import effect, RW_GMEM_ATOMIC_SYNC

        def cg(mem, index, v):
            return Builtin(body="", includes=[], return_val="")

        @builtin(eval_return_type=ll.void, codegen_func=cg, effects=(effect(RW_GMEM_ATOMIC_SYNC, "v"), ))
        def eff_op(mem, index, v):
            raise RuntimeError("should not be called in compilation")

        statics = self._statics(
            """
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    V: ll.int32 = 7
    tid: ll.int32 = ll.threadIdx_x()
    eff_op(q, tid, V)
""", {"eff_op": eff_op})
        self.assertNotIn("V", statics)

    def test_unspelled_parameter_is_unknown_not_pure(self):
        # A parameter with NO declared effect stays conservative even when
        # another parameter carries a declared read-only effect.
        from little_kernel.language.builtin_base import builtin, Builtin
        from little_kernel.core.memory_effects import MemorySpace, r as effect_r

        def cg(a, b):
            return Builtin(body="", includes=[], return_val="")

        @builtin(eval_return_type=ll.void, codegen_func=cg, effects=(effect_r(MemorySpace.GMEM, "a"), ))
        def half_known(a, b):
            raise RuntimeError("should not be called in compilation")

        statics = self._statics(
            """
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    V: ll.int32 = 7
    W: ll.int32 = 9
    half_known(V, W)
""", {"half_known": half_known})
        self.assertIn("V", statics)  # explicit R effect: established read-only
        self.assertNotIn("W", statics)  # unspelled: unknown, conservative

    def test_scalar_local_survives_range_and_alloc_consumers(self):
        # The remaining PROVEN read-only consumers — real builtin ``range``
        # and the allocation-extent primitive — still keep a scalar static.
        code = _emit("""
def kern(q: ll.Tensor[ll.int32]) -> ll.void:
    K: ll.int32 = 7
    tid: ll.int32 = ll.threadIdx_x()
    for s in range(K):
        q[tid] = K
""")
        self.assertIn("static constexpr int32_t K = 7;", code)


class TestRuntimeExtentDiagnosis(unittest.TestCase):
    """Non-constant extents fail early with a located ValueError, not a
    guessed constant and not invalid CUDA."""

    def test_parameter_extent_diagnosed(self):
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    ll.alloc_shared_memory("s", ll.int32, n)
""")
        self.assertIn("non-constant element count", str(cm.exception))
        self.assertIn("line 3", str(cm.exception))

    def test_thread_id_extent_diagnosed(self):
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    ll.alloc_shared_memory("s", ll.int32, tid)
""")
        self.assertIn("non-constant element count", str(cm.exception))

    def test_reassigned_extent_diagnosed(self):
        # A name bound twice is mutated, hence not provably static; using it
        # as an extent must be diagnosed rather than frozen to 8.
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    M: ll.int32 = 8
    M = 16
    ll.alloc_shared_memory("s", ll.int32, M)
""")
        self.assertIn("non-constant element count", str(cm.exception))

    def test_out_of_range_extent_diagnosed(self):
        # 2**40 does not fit int32; the local is not provably static, so the
        # extent is non-constant and must be diagnosed (not truncated).
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    BIG: ll.int32 = 1099511627776
    ll.alloc_shared_memory("s", ll.int32, BIG)
""")
        self.assertIn("non-constant element count", str(cm.exception))

    def test_augmented_assign_mutation_diagnosed(self):
        # A read-modify-write (x += 1) mutates the local; it is not provably
        # static even though there is a single plain assignment first.
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    M: ll.int32 = 8
    M += 4
    ll.alloc_shared_memory("s", ll.int32, M)
""")
        self.assertIn("non-constant element count", str(cm.exception))

    def test_intrinsic_output_then_extent_diagnosed(self):
        # A name written by an intrinsic output lane escapes; using it later
        # as an extent must be diagnosed rather than frozen to its init 0.
        with self.assertRaises(ValueError) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    n: ll.int32 = 0
    r1: ll.int32 = 0
    r2: ll.int32 = 0
    r3: ll.int32 = 0
    tid: ll.int32 = ll.threadIdx_x()
    ll.gmem_ld_v4s32(x, tid * 4, n, r1, r2, r3)
    ll.alloc_shared_memory("s", ll.int32, n)
""")
        self.assertIn("non-constant element count", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
