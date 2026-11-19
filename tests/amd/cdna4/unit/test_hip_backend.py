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
"""HIP backend without a GPU: passes, compile-time unrolling, codegen, AMDGCN intrinsics, kernargs."""

import re
import struct
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.amd.cdna4.codegen.emitter import HipEmitter, codegen_hip
from little_kernel.core.passes import PASSES
from little_kernel.core.type_system import VectorType, get_dtype_size
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR
import little_kernel.backends.amd.cdna4.language.intrin.amdgcn as amdgcn
from little_kernel.backends.amd.cdna4.runtime.hip_runtime import pack_kernel_args, parse_kernel_params

COPIES = 4


@lk.ll_kernel(backend="hip", is_entry=True)
def vec_scale(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], n: ll.int32, s: ll.float32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = x[i] * s


@lk.ll_kernel(backend="hip", is_entry=True)
def unrolled(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    acc = ll.empty([8], dtype=ll.int32, scope="local")
    for i in ll.unroll(range(COPIES)):
        j: ll.int32 = 2 * i + 1  # one value per copy: becomes a literal
        acc[j] = x[tid * 8 + j]
        if i < 2:
            acc[2 * i] = j
        else:
            acc[2 * i] = tid
    for i in ll.unroll(range(2)):
        v: ll.int32 = i
        v = v + tid  # assigned twice: stays a variable, one per copy
        out[v] = i
    total: ll.int32 = 0
    for k in range(8):
        total = total + acc[k]
    out[tid] = total


@lk.ll_kernel(backend="hip", is_entry=True)
def vector_element(out: ll.Tensor[ll.uint32]) -> ll.void:
    words = ll.zeros([1], dtype=ll.uint32x4, scope="local")
    out[ll.threadIdx_x()] = words[0][2] >> 16


@lk.ll_kernel(backend="hip")
def scale_helper(v: ll.float32, s: ll.float32) -> ll.float32:
    return v * s


@lk.ll_kernel(backend="hip", is_entry=True)
def use_helper(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = scale_helper(x[i], 2.0)


def _hip_source(kernel, **codegen_options):
    return kernel.compile(PASSES["hip"], codegen_hip, num_threads=256, disable_var_hoisting=True,
                          codegen_options=codegen_options or None)


@lk.ll_kernel(backend="hip", is_entry=True)
def waitcnt_lit(out: ll.Tensor[ll.int32]) -> ll.void:
    ll.amdgcn_s_waitcnt_vmcnt(16)
    ll.amdgcn_s_waitcnt_lgkmcnt(0)
    out[ll.threadIdx_x()] = 0


@lk.ll_kernel(backend="hip", is_entry=True)
def waitcnt_too_large(out: ll.Tensor[ll.int32]) -> ll.void:
    ll.amdgcn_s_waitcnt_vmcnt(64)
    out[ll.threadIdx_x()] = 0


@lk.ll_kernel(backend="hip", is_entry=True)
def waitcnt_negative(out: ll.Tensor[ll.int32]) -> ll.void:
    ll.amdgcn_s_waitcnt_vmcnt(-1)
    out[ll.threadIdx_x()] = 0


@lk.ll_kernel(backend="hip", is_entry=True)
def waitcnt_runtime(out: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    ll.amdgcn_s_waitcnt_vmcnt(n)
    out[ll.threadIdx_x()] = 0


class TestWaitcntKernelTrace(unittest.TestCase):
    """End-to-end: a kernel calling amdgcn_s_waitcnt_* must emit a validated, exact immediate."""

    def test_literal_immediates_emitted(self):
        code = _hip_source(waitcnt_lit)
        # vmcnt(16): bits 3:0 = 0, bits 15:14 = 1 -> 0x4F70; lgkmcnt(0) -> 0xC07F.
        self.assertIn(f"__builtin_amdgcn_s_waitcnt({0x4F70});", code)
        self.assertIn(f"__builtin_amdgcn_s_waitcnt({0xC07F});", code)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            _hip_source(waitcnt_too_large)
        with self.assertRaises(ValueError):
            _hip_source(waitcnt_negative)

    def test_runtime_value_rejected(self):
        with self.assertRaises(ValueError):
            _hip_source(waitcnt_runtime)


class TestCompileTimeUnroll(unittest.TestCase):
    """PASSES["hip"] expands ll.unroll loops before codegen."""

    @classmethod
    def setUpClass(cls):
        cls.code = _hip_source(unrolled)

    def test_constant_indices(self):
        for j in (1, 3, 5, 7):
            # Both spellings preserve multiplication-before-addition; ordinary
            # codegen may omit the redundant parentheses after the main merge.
            self.assertTrue(f"acc[{j}] = x[((tid * 8) + {j})];" in self.code
                            or f"acc[{j}] = x[tid * 8 + {j}];" in self.code)

    def test_constant_branches_pruned(self):
        self.assertIn("acc[0] = 1;", self.code)
        self.assertIn("acc[2] = 3;", self.code)
        self.assertIn("acc[4] = tid;", self.code)
        self.assertIn("acc[6] = tid;", self.code)
        self.assertNotIn("if ((", self.code)

    def test_propagated_names_disappear(self):
        self.assertNotRegex(self.code, r"\bj(__u\d+)?\b")

    def test_reassigned_locals_renamed_per_copy(self):
        names = set(re.findall(r"\bv__u\d+\b", self.code))
        self.assertEqual(len(names), 2)

    def test_plain_loops_kept(self):
        self.assertIn("for (", self.code)


class TestHipCodegen(unittest.TestCase):

    def test_kernel_signature(self):
        code = _hip_source(vec_scale, waves_per_eu=1)
        self.assertIn("#include <hip/hip_runtime.h>", code)
        self.assertIn(
            'extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256), amdgpu_waves_per_eu(1, 1))) '
            "void vec_scale(float* x, float* out, int32_t n, float s) {", code)

    def test_no_waves_per_eu_by_default(self):
        self.assertNotIn("amdgpu_waves_per_eu", _hip_source(vec_scale))

    def test_vector_element(self):
        self.assertIn("words[0][2] >> 16", _hip_source(vector_element))

    def test_only_entry_has_c_linkage(self):
        """Regression: device helpers keep C++ linkage; extern \"C\" covers only the entry.

        Wrapping __device__ helpers in extern \"C\" gives them unmangled names (ODR
        collisions across translation units) and breaks the C++ templates/overloads
        pulled in from the HIP headers ("templates must have C++ linkage").
        """
        code = _hip_source(use_helper)
        self.assertIn("__device__ __forceinline__ float scale_helper(float v, float s) {", code)
        self.assertIn(
            'extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256))) '
            "void use_helper(float* x, float* out, int32_t n) {", code)
        self.assertEqual(code.count('extern "C"'), 1)
        self.assertNotIn('extern "C" {', code)

    def test_helper_emitted_before_entry_with_cxx_linkage(self):
        code = _hip_source(use_helper)
        helper_def = code.index("__device__ __forceinline__ float scale_helper(")
        entry_def = code.index('extern "C" __global__')
        self.assertLess(helper_def, entry_def)
        everything_before_helper = code[:helper_def]
        self.assertNotIn('extern "C"', everything_before_helper)

    def test_type_mapping(self):
        emitter = HipEmitter(ctx={"ll": ll}, all_types={})
        self.assertEqual(emitter._lltype_to_cpp(ll.bfloat16), "__hip_bfloat16")
        self.assertEqual(emitter._lltype_to_cpp(ll.float16), "__half")
        self.assertEqual(emitter._lltype_to_cpp(ll.int32x4), "lk_int32x4")
        self.assertEqual(emitter._lltype_to_cpp(ll.float32x4), "lk_float32x4")
        self.assertEqual(emitter._lltype_to_cpp(ll.AmdBufferRsrc), "__amdgpu_buffer_rsrc_t")
        self.assertIn("typedef int lk_int32x8 __attribute__((ext_vector_type(8)));", emitter.header_buffer.getvalue())


class TestVectorTypes(unittest.TestCase):

    def test_cached_and_sized(self):
        self.assertIs(VectorType(ll.int32, 4), ll.int32x4)
        self.assertEqual(str(ll.float32x16), "float32x16")
        self.assertEqual(get_dtype_size(ll.int32x8), 32)


def _codegen(intrinsic, *args):
    return getattr(intrinsic, CODEGEN_FUNC_ATTR)(*args)


def _waitcnt_imm(builtin):
    expr = re.fullmatch(r"__builtin_amdgcn_s_waitcnt\((.*)\)", builtin.return_val).group(1)
    return eval(expr)


class TestAmdgcnIntrinsics(unittest.TestCase):

    def test_all_are_builtins(self):
        names = [name for name in dir(amdgcn) if name.startswith("amdgcn_")]
        self.assertGreaterEqual(len(names), 18)
        for name in names:
            self.assertTrue(getattr(getattr(amdgcn, name), BUILTIN_ATTR, False), name)
            self.assertTrue(hasattr(ll, name), name)

    def test_waitcnt_encoding(self):
        # gfx9 s_waitcnt: vmcnt bits 3:0 and 15:14, expcnt 6:4, lgkmcnt 11:8; the other counters at "no wait"
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 12)), 0x0F7C)
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 18)), 0x4F72)
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_lgkmcnt, 0)), 0xC07F)
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_lgkmcnt, 3)), 0xC37F)

    def test_waitcnt_vmcnt_exact_imm(self):
        # Boundary immediates: bits 3:0 hold n % 16, bits 15:14 hold n / 16.
        self.assertEqual(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 0).return_val, "__builtin_amdgcn_s_waitcnt(3952)")
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 15)), 0x0F7F)
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 16)), 0x4F70)
        self.assertEqual(_waitcnt_imm(_codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, 63)), 0xCF7F)

    def test_waitcnt_vmcnt_rejects_out_of_range(self):
        # 64 and -1 were silently masked to the encoding of 0 (with bit 16 set for 64).
        for bad in (-1, 64, 4096):
            with self.assertRaises(ValueError, msg=f"vmcnt({bad})"):
                _codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, bad)
        for bad in (-1, 16):
            with self.assertRaises(ValueError, msg=f"lgkmcnt({bad})"):
                _codegen(amdgcn.amdgcn_s_waitcnt_lgkmcnt, bad)

    def test_waitcnt_rejects_non_constant(self):
        # A runtime operand renders as an identifier; s_waitcnt needs an immediate.
        for arg in ("n", "cnt + 1"):
            with self.assertRaises(ValueError, msg=f"vmcnt({arg})"):
                _codegen(amdgcn.amdgcn_s_waitcnt_vmcnt, arg)
            with self.assertRaises(ValueError, msg=f"lgkmcnt({arg})"):
                _codegen(amdgcn.amdgcn_s_waitcnt_lgkmcnt, arg)

    def test_mfma(self):
        out = _codegen(amdgcn.amdgcn_mfma_scale_f32_16x16x128_f8f6f4, "a", "b", "c", 0, 1, 0, "sa", 2, "sb")
        self.assertEqual(out.return_val,
                         "__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 1, 0, sa, 2, sb)")

    def test_buffer_load_lds(self):
        out = _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", 16, "vo", "so")
        self.assertIn("__builtin_amdgcn_raw_ptr_buffer_load_lds(r, (lk_lds_ptr_t)", out.return_val)
        self.assertTrue(out.return_val.endswith(", 16, vo, so, 0, 0)"))

    def test_barrier_is_plain_s_barrier(self):
        self.assertIn('asm volatile("s_barrier" ::: "memory")', _codegen(amdgcn.amdgcn_s_barrier).return_val)

    def test_buffer_store_b128_keeps_a_wait_state(self):
        out = _codegen(amdgcn.amdgcn_buffer_store_b128, "v", "r", "vo", "so")
        self.assertEqual(out.return_val, "lk_buffer_store_b128((lk_uint32x4)(v), r, vo, so)")
        store = out.body.index("__builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);")
        self.assertGreater(out.body.index('asm volatile("s_nop 0");'), store)

    def test_cvt_pk_bf16_f32x8(self):
        out = _codegen(amdgcn.amdgcn_cvt_pk_bf16_f32x8, "lo", "hi")
        self.assertIn("lk_cvt_pk_bf16_f32", out.body)
        self.assertTrue(out.return_val.startswith("((lk_uint32x4){lk_cvt_pk_bf16_f32((lo)[0], (lo)[1])"))
        self.assertTrue(out.return_val.endswith("lk_cvt_pk_bf16_f32((hi)[2], (hi)[3])})"))


# Kernels for the size_bytes contract tests live at module level: the
# ll_kernel tracer does not capture closure variables, so each kernel spells
# its size as an in-body literal (as the benchmarks do with module constants).
# The invalid ones are never compiled successfully -- the tests only assert
# that compilation rejects them -- but they must trace and type-check first so
# the failure is observed at the intrinsic's codegen validation.


@lk.ll_kernel(backend="hip", is_entry=True)
def _load_dma_size4(x: ll.Tensor[ll.uint8], n: ll.int32) -> ll.void:
    smem = ll.empty([1 << 16], dtype=ll.uint8, scope="dynamic_shared")
    rsrc: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(x, 0, n)
    tid: ll.int32 = ll.threadIdx_x()
    ll.amdgcn_buffer_load_lds(rsrc, smem, 0, 4, tid * 4, 0)
    ll.amdgcn_s_waitcnt_vmcnt(0)


@lk.ll_kernel(backend="hip", is_entry=True)
def _load_dma_size16(x: ll.Tensor[ll.uint8], n: ll.int32) -> ll.void:
    smem = ll.empty([1 << 16], dtype=ll.uint8, scope="dynamic_shared")
    rsrc: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(x, 0, n)
    tid: ll.int32 = ll.threadIdx_x()
    ll.amdgcn_buffer_load_lds(rsrc, smem, 0, 16, tid * 16, 0)
    ll.amdgcn_s_waitcnt_vmcnt(0)


@lk.ll_kernel(backend="hip", is_entry=True)
def _load_dma_size8(x: ll.Tensor[ll.uint8], n: ll.int32) -> ll.void:
    smem = ll.empty([1 << 16], dtype=ll.uint8, scope="dynamic_shared")
    rsrc: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(x, 0, n)
    tid: ll.int32 = ll.threadIdx_x()
    ll.amdgcn_buffer_load_lds(rsrc, smem, 0, 8, tid * 8, 0)


@lk.ll_kernel(backend="hip", is_entry=True)
def _load_dma_size_runtime(x: ll.Tensor[ll.uint8], n: ll.int32) -> ll.void:
    smem = ll.empty([1 << 16], dtype=ll.uint8, scope="dynamic_shared")
    rsrc: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(x, 0, n)
    tid: ll.int32 = ll.threadIdx_x()
    sz: ll.int32 = n  # runtime value: renders as a name, not a literal
    ll.amdgcn_buffer_load_lds(rsrc, smem, 0, sz, tid * 16, 0)


class TestBufferLoadLdsSizeContract(unittest.TestCase):
    """``amdgcn_buffer_load_lds`` documents size_bytes as "4 or 16, a constant".

    Before the fix the value was interpolated verbatim into the clang builtin
    call, so an out-of-contract size (8) or a non-constant expression compiled
    to invalid HIP and only failed inside hipcc.  The codegen wrapper now
    enforces the contract up front: {4, 16} compile, anything else raises a
    frontend ValueError naming the intrinsic before any code is generated.
    """

    def test_size_4_and_16_emit_builtin(self):
        for size in (4, 16):
            out = _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", size, "vo", "so")
            self.assertTrue(out.return_val.endswith(f", {size}, vo, so, 0, 0)"), size)

    def test_emitter_rendered_constant_spellings(self):
        # Through the emitter a constant argument arrives as C++ text (decimal,
        # or hex when the kernel wrote 0x10); these spellings name the same
        # compile-time constants and must keep compiling.
        for spelling, rendered in (("4", 4), ("16", 16), ("0x4", 4), ("0x10", 16)):
            out = _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", spelling, "vo", "so")
            self.assertTrue(out.return_val.endswith(f", {rendered}, vo, so, 0, 0)"), spelling)

    def test_undocumented_size_8_rejected(self):
        # Regression: 8 used to slip into the builtin call verbatim and fail
        # only inside hipcc.  It must now fail in the frontend.
        with self.assertRaisesRegex(ValueError, "amdgcn_buffer_load_lds"):
            _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", 8, "vo", "so")

    def test_other_undocumented_sizes_rejected(self):
        for size in (0, 1, 2, 32, -4):
            with self.assertRaisesRegex(ValueError, "amdgcn_buffer_load_lds"):
                _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", size, "vo", "so")

    def test_non_constant_expression_rejected(self):
        # A runtime value renders as a name/expression, never as an integer
        # literal: it must not reach the builtin call.
        for rendered in ("sz", "n + 1", "(n)"):
            with self.assertRaisesRegex(ValueError, "amdgcn_buffer_load_lds"):
                _codegen(amdgcn.amdgcn_buffer_load_lds, "r", "smem", "off", rendered, "vo", "so")

    def test_end_to_end_compile_accepts_valid_sizes(self):
        # The ll_kernel tracer does not capture closure variables, so the
        # valid-size kernels live at module level with the size as a literal.
        for kernel, size in ((_load_dma_size4, 4), (_load_dma_size16, 16)):
            code = _hip_source(kernel)
            self.assertIn("__builtin_amdgcn_raw_ptr_buffer_load_lds(", code)
            self.assertIn(f", {size}, ", code)

    def test_end_to_end_compile_rejects_size_8_before_codegen(self):
        # The rejection must surface as a frontend compile error naming the
        # intrinsic, never as generated source containing ``..., 8, ...``.
        with self.assertRaisesRegex(ValueError, "amdgcn_buffer_load_lds"):
            _hip_source(_load_dma_size8)

    def test_end_to_end_compile_rejects_non_constant_size(self):
        with self.assertRaisesRegex(ValueError, "amdgcn_buffer_load_lds"):
            _hip_source(_load_dma_size_runtime)


class TestKernelArgs(unittest.TestCase):

    def test_parse_params(self):
        self.assertEqual(parse_kernel_params(_hip_source(vec_scale), "vec_scale"), ["P", "P", "i", "f"])

    def test_parse_multiword_pointer_spellings(self):
        # Pointer-ness must be decided from the whole parameter text, never from
        # the type words left after dropping the final identifier: "long long *p"
        # and "unsigned long *p" lose their "*" under that latter reading and
        # would be misclassified as (unsupported) multi-word scalars.
        code = ("__global__ void ptrs(long long *a, unsigned long *b, long long* c, "
                "unsigned* d, int *e, const float *f, unsigned g, int h) { }")
        self.assertEqual(parse_kernel_params(code, "ptrs"), ["P", "P", "P", "P", "P", "P", "I", "i"])

    def test_parse_rejects_unknown_scalar(self):
        with self.assertRaises(RuntimeError):
            parse_kernel_params("__global__ void bad(long long x) { }", "bad")

    def test_parse_unknown_kernel(self):
        with self.assertRaises(RuntimeError):
            parse_kernel_params("__global__ void present(int x) { }", "absent")

    def test_pack_alignment_and_unsigned(self):
        buf = pack_kernel_args(["i", "P", "I", "q"], [7, 0x1000, -1, -5])
        self.assertEqual(len(buf), 32)
        self.assertEqual(struct.unpack_from("<i", buf, 0)[0], 7)
        self.assertEqual(struct.unpack_from("<Q", buf, 8)[0], 0x1000)
        self.assertEqual(struct.unpack_from("<I", buf, 16)[0], 0xFFFFFFFF)
        self.assertEqual(struct.unpack_from("<q", buf, 24)[0], -5)

    def test_argument_count(self):
        with self.assertRaises(TypeError):
            pack_kernel_args(["i"], [1, 2])


if __name__ == "__main__":
    unittest.main()
