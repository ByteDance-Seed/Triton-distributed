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
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Unit tests for the generic-N WGMMA dispatchers (coverage.mma_pipeline).

wgmma_mma_async_ss / wgmma_mma_async_rs take the tile N from the full PTX
shape whitelist {8, 16, 24, ..., 256} (multiples of 8, f16/bf16 operands,
f32 accumulator) plus +/-1 scale_a/scale_b immediates, closing the gap where
only N in {64, 128, 256} with pinned 1,1 operand scales was reachable.

Legality model (per cw.codegen.tilelang-ptx-inline-assembly): n, dtype, flags
and scales are whitelist-checked BEFORE any asm text is composed; illegal
configurations raise ValueError, never a malformed template.
"""

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma
from little_kernel.language.builtin_base import BUILTIN_ATTR


class TestWgmmaGenericExports(unittest.TestCase):

    def test_dispatchers_are_builtins(self):
        self.assertTrue(getattr(wgmma.wgmma_mma_async_ss, BUILTIN_ATTR, False))
        self.assertTrue(getattr(wgmma.wgmma_mma_async_rs, BUILTIN_ATTR, False))

    def test_ll_namespace_exports(self):
        self.assertIs(ll.wgmma_mma_async_ss, wgmma.wgmma_mma_async_ss)
        self.assertIs(ll.wgmma_mma_async_rs, wgmma.wgmma_mma_async_rs)

    def test_legal_n_whitelist_is_full_ptx_range(self):
        self.assertEqual(wgmma._WGMMA_LEGAL_NCOLS, tuple(range(8, 257, 8)))


class TestWgmmaGenericSsCodegen(unittest.TestCase):

    def test_n8_smallest_tile(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 8)
        self.assertIn("wgmma_mma_ss_n8_bf16_sa1sb1d1ta0tb0_fn(", b.body)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", b.body)
        self.assertEqual(b.body.count('"+f"'), 4)  # n//2 accumulator regs
        self.assertIn('"%4,%5,p,1,1,0,0;\\n}\\n"', b.body)
        self.assertEqual(b.return_val, "wgmma_mma_ss_n8_bf16_sa1sb1d1ta0tb0_fn(acc, da, db)")

    def test_n64_defaults_pin_legacy_operands(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64)
        self.assertIn("m64n64k16.f32.bf16.bf16", b.body)
        self.assertEqual(b.body.count('"+f"'), 32)
        self.assertIn('"%32,%33,p,1,1,0,0;\\n}\\n"', b.body)
        self.assertIn("setp.ne.b32 p, 1, 0;", b.body)

    def test_n192_mid_tile(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 192)
        self.assertIn("m64n192k16.f32.bf16.bf16", b.body)
        self.assertEqual(b.body.count('"+f"'), 96)
        self.assertIn('"%96,%97,p,1,1,0,0;\\n}\\n"', b.body)

    def test_n256_f16(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 256, dtype="f16")
        self.assertIn("m64n256k16.f32.f16.f16", b.body)
        self.assertEqual(b.body.count('"+f"'), 128)
        self.assertIn('"%128,%129,p,1,1,0,0;\\n}\\n"', b.body)

    def test_scale_scale_d_trans_immediates(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_a=-1, scale_b=-1, scale_d=0, trans_a=1,
                                             trans_b=1)
        # Helper name must be a legal C++ identifier: the signed PTX
        # immediate -1 is encoded as the name token "n1", while the PTX
        # operand text below keeps the real signed value -1.
        self.assertIn("wgmma_mma_ss_n64_bf16_san1sbn1d0ta1tb1_fn(", b.body)
        self.assertNotIn("sa-1", b.body)
        self.assertIn("setp.ne.b32 p, 0, 0;", b.body)
        self.assertIn('"%32,%33,p,-1,-1,1,1;\\n}\\n"', b.body)

    def test_runtime_scale_d_register_predicate(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_d=_RuntimeInt())
        self.assertIn("wgmma_mma_ss_n64_bf16_sa1sb1drta0tb0_fn(", b.body)
        self.assertIn(", int scale_d)", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %34, 0;", b.body)

    def test_every_legal_n_emits_n_over_2_accumulators(self):
        for n in wgmma._WGMMA_LEGAL_NCOLS:
            b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", n)
            self.assertEqual(b.body.count('"+f"'), n // 2, n)
            self.assertIn(f"m64n{n}k16.f32.bf16.bf16", b.body)

    def test_distinct_configs_get_distinct_helpers(self):
        names = set()
        for n in (64, 128):
            for dtype in ("bf16", "f16"):
                for sa in (1, -1):
                    b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", n, dtype=dtype, scale_a=sa)
                    helper = b.return_val.split("(")[0]
                    self.assertNotIn(helper, names)
                    names.add(helper)


class TestWgmmaGenericRsCodegen(unittest.TestCase):

    def test_n64_defaults_pin_legacy_operands(self):
        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 64)
        self.assertIn("wgmma_mma_rs_n64_bf16_sa1sb1d1tb0_fn(", b.body)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", b.body)
        self.assertIn('"{%32,%33,%34,%35},%36,p,1,1,0;\\n}\\n"', b.body)

    def test_n128_f16_scale_a_negative(self):
        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, dtype="f16", scale_a=-1)
        self.assertIn("wgmma_mma_rs_n128_f16_san1sb1d1tb0_fn(", b.body)
        self.assertNotIn("sa-1", b.body)
        self.assertIn("m64n128k16.f32.f16.f16", b.body)
        self.assertEqual(b.body.count('"+f"'), 64)
        self.assertIn('"{%64,%65,%66,%67},%68,p,-1,1,0;\\n}\\n"', b.body)

    def test_runtime_scale_d_register_predicate(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, scale_d=_RuntimeInt())
        self.assertIn("wgmma_mma_rs_n128_bf16_sa1sb1drtb0_fn(", b.body)
        self.assertIn(", int scale_d)", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %69, 0;", b.body)


class TestWgmmaGenericHelperNaming(unittest.TestCase):
    """Every supported immediate combination must emit a helper name that is
    both a legal C++ identifier and injective across the configuration space.

    The signed PTX ±1 scale immediates are the risky characters: a raw "-1"
    interpolation produces "...sa-1..." which is not a legal identifier, so
    the name token for -1 is "n1" while the PTX operand text keeps -1.
    """

    _IDENT = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")
    _REPRESENTATIVE_N = (8, 64, 128, 192, 256)
    _RUNTIME = object()  # any non-compile-time-int stands in for a tracer int

    def _ss_configs(self):
        for n in self._REPRESENTATIVE_N:
            for dtype in ("bf16", "f16"):
                for sa in (1, -1):
                    for sb in (1, -1):
                        for sd in (0, 1, self._RUNTIME):
                            for ta in (0, 1):
                                for tb in (0, 1):
                                    yield (("acc", "da", "db", n),
                                           dict(dtype=dtype, scale_a=sa, scale_b=sb, scale_d=sd, trans_a=ta,
                                                trans_b=tb))

    def _rs_configs(self):
        for n in self._REPRESENTATIVE_N:
            for dtype in ("bf16", "f16"):
                for sa in (1, -1):
                    for sb in (1, -1):
                        for sd in (0, 1, self._RUNTIME):
                            for tb in (0, 1):
                                yield (("acc", "a0", "a1", "a2", "a3", "db", n),
                                       dict(dtype=dtype, scale_a=sa, scale_b=sb, scale_d=sd, trans_b=tb))

    def _check_names(self, gen, configs):
        names = {}
        for args, kwargs in configs:
            b = gen(*args, **kwargs)
            name = b.return_val.split("(")[0]
            self.assertTrue(name.endswith("_fn"), name)
            self.assertRegex(name, self._IDENT, name)
            clash = names.setdefault(name, (tuple(args[3:]) + tuple(sorted(kwargs.items()))))
            self.assertEqual(clash, (tuple(args[3:]) + tuple(sorted(kwargs.items()))),
                             f"non-injective helper name {name}")
        return names

    def test_ss_helper_names_legal_and_injective(self):
        names = self._check_names(wgmma.codegen_wgmma_mma_async_ss, list(self._ss_configs()))
        # 5 N x 2 dtype x 2 sa x 2 sb x 3 scale_d x 2 ta x 2 tb
        self.assertEqual(len(names), 5 * 2 * 2 * 2 * 3 * 2 * 2)

    def test_rs_helper_names_legal_and_injective(self):
        names = self._check_names(wgmma.codegen_wgmma_mma_async_rs, list(self._rs_configs()))
        # 5 N x 2 dtype x 2 sa x 2 sb x 3 scale_d x 2 tb
        self.assertEqual(len(names), 5 * 2 * 2 * 2 * 3 * 2)

    def test_negative_scale_ptx_immediate_preserved(self):
        """The name encoding must not leak into the PTX operand values."""
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_a=-1, scale_b=-1, scale_d=0)
        self.assertIn("san1sbn1", b.return_val)
        self.assertIn('"%32,%33,p,-1,-1,0,0;\\n}\\n"', b.body)

    def test_scale_token_validation(self):
        self.assertEqual(wgmma._wgmma_scale_token(1), "1")
        self.assertEqual(wgmma._wgmma_scale_token(-1), "n1")
        with self.assertRaises(ValueError):
            wgmma._wgmma_scale_token(0)


_NVCC = shutil.which("nvcc")


@unittest.skipIf(_NVCC is None, "nvcc not available in this environment")
class TestWgmmaGenericNvccObjectCompile(unittest.TestCase):
    """noGPU object-compilation regression: the emitted helper for a
    representative negative-scale configuration (plus a positive-scale
    compatibility check) must compile with the pinned NVCC/SM90a flags.
    Only object compilation is exercised here; GPU execution stays with the
    trusted validation queue."""

    def _object_compile(self, body):
        src = textwrap.dedent("""\
            #include <cstdint>
            """)
        src += body
        with tempfile.TemporaryDirectory() as td:
            cu = os.path.join(td, "wgmma_helper.cu")
            obj = os.path.join(td, "wgmma_helper.o")
            with open(cu, "w") as f:
                f.write(src)
            cmd = [_NVCC, "-c", "-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "-o", obj, cu]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"nvcc {' '.join(cmd)}\nstdout:\n{proc.stdout}\n"
                             f"stderr:\n{proc.stderr}")
            self.assertTrue(os.path.getsize(obj) > 0)

    def test_ss_negative_scales_object_compile(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype="bf16", scale_a=-1, scale_b=-1, scale_d=0,
                                             trans_a=1, trans_b=1)
        self.assertIn("wgmma_mma_ss_n64_bf16_san1sbn1d0ta1tb1_fn", b.body)
        self._object_compile(b.body)

    def test_ss_positive_scales_object_compile(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64)
        self.assertIn("wgmma_mma_ss_n64_bf16_sa1sb1d1ta0tb0_fn", b.body)
        self._object_compile(b.body)

    def test_rs_negative_scale_object_compile(self):
        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, dtype="f16", scale_a=-1,
                                             scale_b=-1, scale_d=1, trans_b=1)
        self.assertIn("wgmma_mma_rs_n128_f16_san1sbn1d1tb1_fn", b.body)
        self._object_compile(b.body)

    def test_rs_runtime_scale_d_object_compile(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 64, scale_d=_RuntimeInt())
        self.assertIn("wgmma_mma_rs_n64_bf16_sa1sb1drtb0_fn", b.body)
        self._object_compile(b.body)


class TestWgmmaGenericLegality(unittest.TestCase):
    """Counterexamples: illegal shape/dtype/scale/flag must raise before any
    asm text is composed."""

    def test_illegal_n_values_rejected(self):
        for bad in (0, 4, 7, 9, 100, 264, -8, 8.5):
            with self.subTest(n=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", bad)

    def test_runtime_n_rejected(self):

        class _RuntimeInt:
            pass

        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", _RuntimeInt())

    def test_dtype_whitelist_enforced(self):
        # int8/uint8 are legal (integer m64nNk32 family) and e4m3/e5m2 are
        # legal (FP8 m64nNk32.f32 family); only dtype tokens outside all
        # whitelists must raise. Note "fp8" alone is NOT an FP8 alias: the
        # concrete encoding (e4m3 or e5m2) is required.
        for bad in ("tf32", "f32", "fp8", "fp8_e4m5"):
            with self.subTest(dtype=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 64, dtype=bad)
            with self.subTest(dtype=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype=bad)

    def test_scale_immediates_are_signed_one_only(self):
        for bad in (0, 2, -2, "x"):
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_a=bad)
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_b=bad)

    def test_trans_flags_binary(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, trans_a=2)
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 64, trans_b=-1)

    def test_scale_d_immediate_binary(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_d=3)


class TestWgmmaGenericAgreesWithFixedShapes(unittest.TestCase):
    """Generic dispatch at n=64 with defaults must compose the same opcode
    and operand text as the existing n64 parameterized family (helper names
    differ; PTX payload is identical)."""

    def test_ss_agreement(self):
        generic = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64)
        fixed = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db")
        for token in ("m64n64k16.f32.bf16.bf16", '"%32,%33,p,1,1,0,0;', "setp.ne.b32 p, 1, 0;"):
            self.assertIn(token, generic.body)
            self.assertIn(token, fixed.body)

    def test_rs_agreement(self):
        generic = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 64)
        fixed = wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db")
        for token in ("m64n64k16.f32.bf16.bf16", '"{%32,%33,%34,%35},%36,p,1,1,0;'):
            self.assertIn(token, generic.body)
            self.assertIn(token, fixed.body)

    def test_fixed_shape_templates_untouched(self):
        """The pre-existing fixed-shape intrinsics stayed byte-identical."""
        b = wgmma.codegen_wgmma_m64n64k16("acc", "da", "db")
        self.assertEqual(b.return_val, 'wgmma_m64n64k16_fn(acc, da, db)')
        self.assertIn('"%32,%33,p,1,1,0,0;\\n}\\n"', b.body)


class TestWgmmaIntDtypeRouting(unittest.TestCase):
    """Integer operand dtypes (s8/u8) route the generic dispatchers to the
    m64nNk32.s32.{s8,u8}.{s8,u8} family instead of the float k16 whitelist."""

    def test_aliases_normalize_to_int_family(self):
        for alias, ptx in (("s8", "s8"), ("int8", "s8"), ("int8_t", "s8"), ("u8", "u8"), ("uint8", "u8"), ("uint8_t",
                                                                                                           "u8")):
            with self.subTest(alias=alias):
                b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype=alias)
                self.assertIn(f"k32.s32.{ptx}.{ptx}", b.body)

    def test_float_dtypes_do_not_route_int(self):
        for dtype in ("bf16", "f16", "half", "float16", "bfloat16", "fp16"):
            with self.subTest(dtype=dtype):
                b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype=dtype)
                self.assertIn("k16.f32.", b.body)
                self.assertNotIn("k32.s32.", b.body)

    def test_unknown_int_alias_rejected(self):
        # e4m3/e5m2 route to the legal FP8 m64nNk32.f32 family, so they are
        # no longer rejected here; every other non-int, non-float token still
        # is ("fp8" without a concrete encoding remains illegal).
        for bad in ("fp8", "tf32", "s16", "int32", "f32"):
            with self.subTest(dtype=bad), self.assertRaises(ValueError):
                wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype=bad)


class TestWgmmaIntSsCodegen(unittest.TestCase):
    """m64nNk32.s32.s8.s8 / u8.u8 SS codegen (n//2 int32 accumulators)."""

    def test_n128_s8_exact_teacher_opcode(self):
        """The reproducer configuration: m64n128k32.s32.s8.s8 with scale_d=0
        overwrite, matching the WGMMA+TMA integer GEMM teacher."""
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_a=1, scale_b=1, scale_d=0,
                                             trans_a=0, trans_b=0)
        self.assertEqual(b.return_val, "wgmma_mma_ss_n128k32_s32_s8_s8_d0_fn(acc, da, db)")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8", b.body)
        self.assertIn("int32_t* c", b.body)
        self.assertIn("<cstdint>", b.includes)
        self.assertEqual(b.body.count('"+r"'), 64)  # n//2 int32 regs
        # No scale/trans immediates on the integer opcode: operands end at p.
        self.assertIn('"%64,%65,p;\\n}\\n"', b.body)
        # scale_d=0 predicate (overwrite).
        self.assertIn("setp.ne.b32 p, 0, 0;", b.body)

    def test_accumulate_predicate_scale_d_1(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_d=1)
        self.assertIn("wgmma_mma_ss_n128k32_s32_s8_s8_d1_fn(", b.body)
        self.assertIn("setp.ne.b32 p, 1, 0;", b.body)

    def test_u8_variant(self):
        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, dtype="u8", scale_d=1)
        self.assertIn("m64n64k32.s32.u8.u8", b.body)
        self.assertIn("wgmma_mma_ss_n64k32_s32_u8_u8_d1_fn(", b.body)

    def test_every_legal_n_emits_n_over_2_int32_accumulators(self):
        for n in wgmma._WGMMA_LEGAL_NCOLS:
            b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", n, dtype="s8")
            self.assertEqual(b.body.count('"+r"'), n // 2, n)
            self.assertIn(f"m64n{n}k32.s32.s8.s8", b.body)

    def test_runtime_scale_d_register_predicate(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_d=_RuntimeInt())
        self.assertIn("wgmma_mma_ss_n128k32_s32_s8_s8_dr_fn(", b.body)
        self.assertIn(", int scale_d)", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %66, 0;", b.body)
        # The call forwards the traced value for the scale_d parameter.
        self.assertTrue(b.return_val.startswith("wgmma_mma_ss_n128k32_s32_s8_s8_dr_fn(acc, da, db, "))

    def test_distinct_configs_get_distinct_helpers(self):
        names = set()
        for n in (64, 128, 256):
            for ptx in ("s8", "u8"):
                for sd in (0, 1):
                    b = wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", n, dtype=ptx, scale_d=sd)
                    helper = b.return_val.split("(")[0]
                    self.assertNotIn(helper, names)
                    names.add(helper)


class TestWgmmaIntRsCodegen(unittest.TestCase):
    """m64nNk32.s32.{s8,u8}.{s8,u8} RS codegen (A in registers)."""

    def test_n128_s8(self):
        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, dtype="s8", scale_d=1)
        self.assertEqual(b.return_val, "wgmma_mma_rs_n128k32_s32_s8_s8_d1_fn(acc, a0, a1, a2, a3, db)")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8", b.body)
        self.assertIn("int32_t* c", b.body)
        self.assertEqual(b.body.count('"+r"'), 64)
        # RS operand text: {a0..a3}, desc_b, p then terminates.
        self.assertIn('"{%64,%65,%66,%67},%68,p;\\n}\\n"', b.body)

    def test_runtime_scale_d_register_predicate(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, dtype="u8",
                                             scale_d=_RuntimeInt())
        self.assertIn("wgmma_mma_rs_n128k32_s32_u8_u8_dr_fn(", b.body)
        self.assertIn(", int scale_d)", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %69, 0;", b.body)


class TestWgmmaIntLegality(unittest.TestCase):
    """Integer wgmma has no scale/trans immediates and a k32 shape: counter-
    examples must raise ValueError before any asm text is composed."""

    def test_negative_scales_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_a=-1)
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_b=-1)

    def test_trans_immediates_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", trans_a=1)
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", trans_b=1)
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_rs("acc", "a0", "a1", "a2", "a3", "db", 128, dtype="s8", trans_b=1)

    def test_illegal_n_mentions_k32_family(self):
        with self.assertRaises(ValueError) as cm:
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 7, dtype="s8")
        self.assertIn("m64nNk32", str(cm.exception))

    def test_scale_d_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_mma_async_ss("acc", "da", "db", 128, dtype="s8", scale_d=2)


class TestWgmmaIntEndToEnd(unittest.TestCase):
    """End-to-end: a frontend kernel calling ll.wgmma_mma_async_ss with an
    int8 operand dtype and an int32 local accumulator lowers through the CUDA
    codegen path to the integer WGMMA opcode (no GPU execution)."""

    def test_s8_ss_kernel_emits_int_opcode(self):
        import little_kernel as lk
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def gemm_int8_mma(lo_in: ll.Tensor[ll.uint32]) -> ll.void:
            lo: ll.uint32 = lo_in[0]
            hi: ll.uint32 = lo_in[1]
            desc_a: ll.uint64 = ll.make_desc(lo, hi)
            desc_b: ll.uint64 = ll.make_desc(lo, hi)
            acc = ll.empty([64], dtype=ll.int32, scope="local")
            ll.wgmma_mma_async_ss(acc, desc_a, desc_b, 128, dtype="s8", scale_a=1, scale_b=1, scale_d=0, trans_a=0,
                                  trans_b=0)

        code = gemm_int8_mma.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8", code)
        self.assertIn("int32_t* c", code)
        self.assertIn("wgmma_mma_ss_n128k32_s32_s8_s8_d0_fn(", code)

    def test_float_kernel_still_emits_float_opcode(self):
        import little_kernel as lk
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def gemm_bf16_mma(lo_in: ll.Tensor[ll.uint32]) -> ll.void:
            lo: ll.uint32 = lo_in[0]
            hi: ll.uint32 = lo_in[1]
            desc_a: ll.uint64 = ll.make_desc(lo, hi)
            desc_b: ll.uint64 = ll.make_desc(lo, hi)
            acc = ll.empty([64], dtype=ll.float32, scope="local")
            ll.wgmma_mma_async_ss(acc, desc_a, desc_b, 128, dtype="bf16")

        code = gemm_bf16_mma.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", code)
        self.assertNotIn("k32.s32", code)


if __name__ == "__main__":
    unittest.main()
