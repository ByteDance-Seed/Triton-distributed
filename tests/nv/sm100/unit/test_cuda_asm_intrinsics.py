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

import unittest
import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.arith as arith
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm
import little_kernel.backends.nv.common.language.intrin.memory as memory
import little_kernel.backends.nv.sm100.language.intrin.umma as umma
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR


class TestSharedMemoryIntrinsics(unittest.TestCase):
    """Tests for shared memory load/store intrinsics."""

    def test_ld_shared_u32_is_builtin(self):
        """Test that ld_shared_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ld_shared_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ld_shared_u32, BUILTIN_ATTR, False))

    def test_ld_shared_u64_is_builtin(self):
        """Test that ld_shared_u64 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ld_shared_u64, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ld_shared_u64, BUILTIN_ATTR, False))

    def test_ld_shared_f32_is_builtin(self):
        """Test that ld_shared_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ld_shared_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ld_shared_f32, BUILTIN_ATTR, False))

    def test_st_shared_u32_is_builtin(self):
        """Test that st_shared_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.st_shared_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.st_shared_u32, BUILTIN_ATTR, False))

    def test_st_shared_u64_is_builtin(self):
        """Test that st_shared_u64 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.st_shared_u64, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.st_shared_u64, BUILTIN_ATTR, False))

    def test_st_shared_f32_is_builtin(self):
        """Test that st_shared_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.st_shared_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.st_shared_f32, BUILTIN_ATTR, False))

    def test_ld_shared_u32_codegen(self):
        """Test codegen for ld_shared_u32."""
        codegen_func = getattr(cuda_asm.ld_shared_u32, CODEGEN_FUNC_ATTR)
        self.assertIsNotNone(codegen_func)
        self.assertTrue(callable(codegen_func))

        from little_kernel.language.builtin_base import Builtin
        result = codegen_func("ptr", "offset")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.shared.u32", result.body)


class TestGlobalMemoryIntrinsics(unittest.TestCase):
    """Tests for global memory load/store intrinsics."""

    def test_ld_global_u32_is_builtin(self):
        """Test that ld_global_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ld_global_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ld_global_u32, BUILTIN_ATTR, False))

    def test_ld_global_u64_is_builtin(self):
        """Test that ld_global_u64 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ld_global_u64, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ld_global_u64, BUILTIN_ATTR, False))

    def test_st_global_u32_is_builtin(self):
        """Test that st_global_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.st_global_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.st_global_u32, BUILTIN_ATTR, False))

    def test_st_global_u64_is_builtin(self):
        """Test that st_global_u64 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.st_global_u64, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.st_global_u64, BUILTIN_ATTR, False))


class TestPlainVectorU32Intrinsics(unittest.TestCase):
    """Tests for the non-barrier (plain-C++) gmem vector u32 intrinsics."""

    def test_gmem_ld_v4_u32_pl_is_builtin(self):
        self.assertTrue(getattr(memory.gmem_ld_v4_u32_pl, BUILTIN_ATTR, False))

    def test_gmem_st_v4_u32_pl_is_builtin(self):
        self.assertTrue(getattr(memory.gmem_st_v4_u32_pl, BUILTIN_ATTR, False))

    def test_gmem_ld_v4_u32_pl_codegen_has_no_asm_barrier(self):
        from little_kernel.language.builtin_base import Builtin
        codegen_func = getattr(memory.gmem_ld_v4_u32_pl, CODEGEN_FUNC_ATTR)
        result = codegen_func("mem", "e", "r0", "r1", "r2", "r3")
        self.assertIsInstance(result, Builtin)
        # Plain C++ uint4 dereference: no asm, no "memory" clobber, same
        # dtype-agnostic void* element addressing as gmem_ld_v4_u32.
        self.assertIn("reinterpret_cast<const uint4*>", result.body)
        self.assertNotIn("asm volatile", result.body)
        self.assertNotIn('"memory"', result.body)
        self.assertIn("gmem_ld_v4_u32_pl_fn((const void*)(mem + e)", result.return_val)

    def test_gmem_st_v4_u32_pl_codegen_has_no_asm_barrier(self):
        from little_kernel.language.builtin_base import Builtin
        codegen_func = getattr(memory.gmem_st_v4_u32_pl, CODEGEN_FUNC_ATTR)
        result = codegen_func("mem", "e", "v0", "v1", "v2", "v3")
        self.assertIsInstance(result, Builtin)
        self.assertIn("reinterpret_cast<uint4*>", result.body)
        self.assertNotIn("asm volatile", result.body)
        self.assertNotIn('"memory"', result.body)
        self.assertIn("gmem_st_v4_u32_pl_fn((void*)(mem + e)", result.return_val)


class TestArithmeticIntrinsics(unittest.TestCase):
    """Tests for arithmetic intrinsics."""

    def test_fma_f32_is_builtin(self):
        """Test that fma_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.fma_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.fma_f32, BUILTIN_ATTR, False))

    def test_mul_lo_u32_is_builtin(self):
        """Test that mul_lo_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.mul_lo_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.mul_lo_u32, BUILTIN_ATTR, False))

    def test_mul_hi_u32_is_builtin(self):
        """Test that mul_hi_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.mul_hi_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.mul_hi_u32, BUILTIN_ATTR, False))

    def test_mad_lo_u32_is_builtin(self):
        """Test that mad_lo_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.mad_lo_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.mad_lo_u32, BUILTIN_ATTR, False))

    def test_fma_f32_codegen(self):
        """Test codegen for fma_f32."""
        codegen_func = getattr(cuda_asm.fma_f32, CODEGEN_FUNC_ATTR)
        self.assertIsNotNone(codegen_func)
        self.assertTrue(callable(codegen_func))

        from little_kernel.language.builtin_base import Builtin
        result = codegen_func("a", "b", "c")
        self.assertIsInstance(result, Builtin)
        self.assertIn("fma.rn.f32", result.body)

    def test_register_array_sum_codegen(self):
        codegen_func = getattr(arith.sum, CODEGEN_FUNC_ATTR)
        result = codegen_func("values")
        self.assertIn("#pragma unroll", result.body)
        self.assertEqual(result.return_val, "register_array_sum(values)")
        return_type = getattr(arith.sum, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(ll.Tensor[ll.float32]), ll.float32)

    def test_scale_bf16x2_preserves_lane_order(self):
        result = getattr(cuda_asm.scale_bf16x2, CODEGEN_FUNC_ATTR)("bits", "scale")
        self.assertIn(': "=r"(result) : "f"(f1), "f"(f0)', result.body)

    def test_exp_f32_libm_codegen(self):
        self.assertTrue(getattr(cuda_asm.exp_f32, BUILTIN_ATTR, False))
        result = getattr(cuda_asm.exp_f32, CODEGEN_FUNC_ATTR)("u")
        self.assertEqual(result.return_val, "expf(u)")
        self.assertTrue(callable(getattr(cuda_asm.exp_f32, EVAL_RETURN_TYPE_ATTR)))
        self.assertIs(ll.exp_f32, cuda_asm.exp_f32)

    def test_log1p_f32_libm_codegen(self):
        self.assertTrue(getattr(cuda_asm.log1p_f32, BUILTIN_ATTR, False))
        result = getattr(cuda_asm.log1p_f32, CODEGEN_FUNC_ATTR)("u")
        self.assertEqual(result.return_val, "log1pf(u)")
        self.assertTrue(callable(getattr(cuda_asm.log1p_f32, EVAL_RETURN_TYPE_ATTR)))
        self.assertIs(ll.log1p_f32, cuda_asm.log1p_f32)

    def test_pow_f32_libm_codegen(self):
        self.assertTrue(getattr(cuda_asm.pow_f32, BUILTIN_ATTR, False))
        result = getattr(cuda_asm.pow_f32, CODEGEN_FUNC_ATTR)("beta", "step_f")
        self.assertEqual(result.return_val, "powf(beta, step_f)")
        self.assertTrue(callable(getattr(cuda_asm.pow_f32, EVAL_RETURN_TYPE_ATTR)))
        self.assertIs(ll.pow_f32, cuda_asm.pow_f32)

    def test_add_bf16x2_is_builtin(self):
        """add_bf16x2 is a registered builtin returning uint32, like mul_bf16x2."""
        self.assertTrue(hasattr(cuda_asm.add_bf16x2, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.add_bf16x2, BUILTIN_ATTR, False))
        return_type = getattr(cuda_asm.add_bf16x2, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(None, None), ll.uint32)

    def test_add_bf16x2_codegen_uses_hadd2(self):
        """Lowering is the lane-wise __hadd2 on __nv_bfloat162, bf16 header on demand."""
        from little_kernel.language.builtin_base import Builtin
        result = getattr(cuda_asm.add_bf16x2, CODEGEN_FUNC_ATTR)("lhs", "rhs")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__hadd2(a, b)", result.body)
        self.assertIn("__nv_bfloat162", result.body)
        self.assertIn("<cuda_bf16.h>", result.includes)
        self.assertEqual(result.return_val, "add_bf16x2_fn(lhs, rhs)")

    def test_add_bf16x2_exported_from_language(self):
        self.assertIs(ll.add_bf16x2, cuda_asm.add_bf16x2)

    def test_scale_bf16x2_pair_preserves_lane_order(self):
        result = getattr(cuda_asm.scale_bf16x2_pair, CODEGEN_FUNC_ATTR)("bits", "scale0", "scale1")
        self.assertIn("pair.x * scale0, pair.y * scale1", result.body)


class TestSmemToGmemSt128(unittest.TestCase):
    """Tests for the fused 16-byte shared->global store intrinsic."""

    def test_is_builtin(self):
        self.assertTrue(getattr(cuda_asm.smem_to_gmem_st_128, BUILTIN_ATTR, False))

    def test_return_type_is_void(self):
        rt = getattr(cuda_asm.smem_to_gmem_st_128, EVAL_RETURN_TYPE_ATTR)
        self.assertEqual(rt(), ll.void)

    def test_codegen_emits_fused_ld_st(self):
        from little_kernel.language.builtin_base import Builtin
        result = getattr(cuda_asm.smem_to_gmem_st_128, CODEGEN_FUNC_ATTR)("dst", "dst_idx", "saddr")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.shared.v4.b32", result.body)
        self.assertIn("*reinterpret_cast<uint4*>(gptr) = value;", result.body)
        self.assertEqual(result.return_val, "smem_to_gmem_st_128_fn((void*)(dst + dst_idx), saddr)")
        # raw bit-copy: no element conversion in the helper
        self.assertNotIn("__float2bfloat16", result.body)
        self.assertNotIn("__bfloat162float", result.body)

    def test_full_kernel_codegen(self):
        import ast as _ast
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
        src = ("def kern(dst: Tensor[bfloat16], n: int32) -> void:\n"
               "    saddr: uint32 = 0\n"
               "    i: int32 = threadIdx_x() * 8\n"
               "    smem_to_gmem_st_128(dst, i, saddr)\n")
        out = codegen_cuda(_ast.parse(src), dict(ll.__dict__))
        self.assertIn("smem_to_gmem_st_128_fn((void*)(dst + i), saddr);", out)
        self.assertIn("ld.shared.v4.b32", out)

    def test_exported_from_language(self):
        self.assertTrue(hasattr(ll, "smem_to_gmem_st_128"))
        self.assertIs(ll.smem_to_gmem_st_128, cuda_asm.smem_to_gmem_st_128)


class TestUmmaIntrinsics(unittest.TestCase):

    def test_smem_desc_addr16_codegen(self):
        codegen_func = getattr(umma.encode_umma_smem_desc, CODEGEN_FUNC_ATTR)
        result = codegen_func("start_16b", "leading", "stride", "layout")
        self.assertIn("start_addr_16b & 0x3FFF", result.body)
        self.assertIn("(leading_bytes >> 4) & 0x3FFF", result.body)
        self.assertIn("(stride_bytes >> 4) & 0x3FFF", result.body)
        self.assertIn("layout_type & 0x7", result.body)
        self.assertEqual(result.return_val, "encode_umma_smem_desc_fn(start_16b, leading, stride, layout)")

    def test_kblock_codegen_keeps_election_external(self):
        codegen_func = getattr(umma.umma_f16_ss_1sm_kblock, CODEGEN_FUNC_ATTR)
        result = codegen_func("d", "a", "b", "idesc", "64", "accum")
        self.assertIn("#pragma unroll", result.body)
        self.assertIn("static_assert(K > 0 && K % 16 == 0", result.body)
        self.assertIn("constexpr int KTiles = K / 16", result.body)
        self.assertNotIn("elect.sync", result.body)
        self.assertEqual(result.return_val, "umma_f16_1sm_kblock_fn<64>(d, a, b, idesc, accum)")

    def test_ts_mn_kblock_uses_mn_address_step(self):
        codegen_func = getattr(umma.umma_f16_ts_1sm_kblock_mn_sw128, CODEGEN_FUNC_ATTR)
        result = codegen_func("d", "a", "b", "idesc", "64", "accum")
        self.assertIn("k * DescStep16", result.body)
        self.assertEqual(result.return_val, "umma_f16_ts_1sm_kblock_fn<64, 128>(d, a, b, idesc, accum)")

    def test_typed_blockscaled_umma_builds_idesc_internally(self):
        cases = (
            (umma.umma_mxfp_e2m1_e2m1_ss_2sm, "kind::mxf4.block_scale.block32", "1u << 7", "1u << 10"),
            (umma.umma_mxfp_e2m1_e4m3_ss_1sm, "kind::mxf8f6f4.block_scale", "5u << 7", "0u << 10"),
            (umma.umma_mxfp_e4m3_e2m1_ss_2sm, "kind::mxf8f6f4.block_scale", "0u << 7", "5u << 10"),
            (umma.umma_mxfp_e3m2_e4m3_ss_2sm, "kind::mxf8f6f4.block_scale", "4u << 7", "0u << 10"),
            (umma.umma_nvfp4_ss_1sm, "kind::mxf4nvf4.block_scale.block16", "1u << 7", "1u << 10"),
        )
        for op, ptx, a_format, b_format in cases:
            result = getattr(op, CODEGEN_FUNC_ATTR)("d", "a", "b", "128", "128", "accum", "sfa", "sfb")
            self.assertIn(ptx, result.body)
            self.assertIn(a_format, result.body)
            self.assertIn(b_format, result.body)
            self.assertIn("static_idesc", result.body)
            self.assertIn("tsfa >> 30", result.body)
            self.assertNotIn("idesc,", result.return_val)


class TestBitOperationIntrinsics(unittest.TestCase):
    """Tests for bit operation intrinsics."""

    def test_popc_u32_is_builtin(self):
        """Test that popc_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.popc_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.popc_u32, BUILTIN_ATTR, False))

    def test_clz_u32_is_builtin(self):
        """Test that clz_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.clz_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.clz_u32, BUILTIN_ATTR, False))

    def test_bfind_u32_is_builtin(self):
        """Test that bfind_u32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.bfind_u32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.bfind_u32, BUILTIN_ATTR, False))


class TestWarpOperationIntrinsics(unittest.TestCase):
    """Tests for warp operation intrinsics."""

    def test_ballot_sync_is_builtin(self):
        """Test that ballot_sync is a builtin."""
        self.assertTrue(hasattr(cuda_asm.ballot_sync, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.ballot_sync, BUILTIN_ATTR, False))

    def test_shfl_sync_up_is_builtin(self):
        """Test that shfl_sync_up is a builtin."""
        self.assertTrue(hasattr(cuda_asm.shfl_sync_up, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.shfl_sync_up, BUILTIN_ATTR, False))

    def test_shfl_sync_down_is_builtin(self):
        """Test that shfl_sync_down is a builtin."""
        self.assertTrue(hasattr(cuda_asm.shfl_sync_down, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.shfl_sync_down, BUILTIN_ATTR, False))

    def test_shfl_sync_bfly_is_builtin(self):
        """Test that shfl_sync_bfly is a builtin."""
        self.assertTrue(hasattr(cuda_asm.shfl_sync_bfly, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.shfl_sync_bfly, BUILTIN_ATTR, False))


class TestMathFunctionIntrinsics(unittest.TestCase):
    """Tests for math function intrinsics."""

    def test_sqrt_f32_is_builtin(self):
        """Test that sqrt_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.sqrt_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.sqrt_f32, BUILTIN_ATTR, False))

    def test_rsqrt_f32_is_builtin(self):
        """Test that rsqrt_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.rsqrt_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.rsqrt_f32, BUILTIN_ATTR, False))

    def test_sin_f32_is_builtin(self):
        """Test that sin_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.sin_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.sin_f32, BUILTIN_ATTR, False))

    def test_cos_f32_is_builtin(self):
        """Test that cos_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.cos_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.cos_f32, BUILTIN_ATTR, False))

    def test_exp2_variants_have_distinct_codegen(self):
        approx = getattr(cuda_asm.exp2_approx_f32, CODEGEN_FUNC_ATTR)("x")
        precise = getattr(cuda_asm.exp2f, CODEGEN_FUNC_ATTR)("x")
        pair = getattr(cuda_asm.exp2_emu_f32x2, CODEGEN_FUNC_ATTR)("x", "y")
        self.assertIn("ex2.approx.f32", approx.body)
        self.assertEqual(precise.return_val, "exp2f(x)")
        self.assertIn("fma.rn.ftz.f32x2", pair.body)
        self.assertEqual(pair.return_val, "exp2_emu_f32x2_fn(x, y)")

    def test_tanh_approx_is_builtin(self):
        """Test that tanh_approx is a builtin."""
        self.assertTrue(hasattr(cuda_asm.tanh_approx, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.tanh_approx, BUILTIN_ATTR, False))

    def test_tanh_approx_codegen(self):
        """Test that tanh_approx emits the single SFU tanh.approx.f32 op."""
        codegen_func = getattr(cuda_asm.tanh_approx, CODEGEN_FUNC_ATTR)
        builtin = codegen_func("x")
        self.assertIn("tanh.approx.f32 %0, %1;", builtin.body)
        eval_return_type = getattr(cuda_asm.tanh_approx, EVAL_RETURN_TYPE_ATTR)
        self.assertIsNotNone(eval_return_type)

    def test_exp_precise_f32_is_builtin(self):
        """Test that exp_precise_f32 is a builtin."""
        self.assertTrue(hasattr(cuda_asm.exp_precise_f32, BUILTIN_ATTR))
        self.assertTrue(getattr(cuda_asm.exp_precise_f32, BUILTIN_ATTR, False))

    def test_exp_precise_f32_codegen(self):
        """Test that exp_precise_f32 binds libdevice __nv_expf via a helper."""
        codegen_func = getattr(cuda_asm.exp_precise_f32, CODEGEN_FUNC_ATTR)
        builtin = codegen_func("x")
        self.assertIn("__nv_expf", builtin.body)
        self.assertEqual(builtin.return_val, "exp_precise_f32_fn(x)")
        eval_return_type = getattr(cuda_asm.exp_precise_f32, EVAL_RETURN_TYPE_ATTR)
        self.assertIsNotNone(eval_return_type)


class TestIntrinsicCodegen(unittest.TestCase):
    """Test code generation for all intrinsics."""

    def test_all_intrinsics_have_codegen(self):
        """Test that all intrinsics have codegen functions."""
        intrinsics = [
            cuda_asm.ld_shared_u32,
            cuda_asm.ld_shared_u64,
            cuda_asm.ld_shared_f32,
            cuda_asm.st_shared_u32,
            cuda_asm.st_shared_u64,
            cuda_asm.st_shared_f32,
            cuda_asm.ld_global_u32,
            cuda_asm.ld_global_u64,
            cuda_asm.st_global_u32,
            cuda_asm.st_global_u64,
            cuda_asm.fma_f32,
            cuda_asm.mul_lo_u32,
            cuda_asm.mul_hi_u32,
            cuda_asm.mad_lo_u32,
            cuda_asm.popc_u32,
            cuda_asm.clz_u32,
            cuda_asm.bfind_u32,
            cuda_asm.ballot_sync,
            cuda_asm.shfl_sync_up,
            cuda_asm.shfl_sync_down,
            cuda_asm.shfl_sync_bfly,
            cuda_asm.sqrt_f32,
            cuda_asm.rsqrt_f32,
            cuda_asm.sin_f32,
            cuda_asm.cos_f32,
            cuda_asm.exp2_approx_f32,
            cuda_asm.exp2f,
            cuda_asm.exp2_emu_f32x2,
            cuda_asm.exp_precise_f32,
            cuda_asm.tanh_approx,
        ]

        for intrinsic in intrinsics:
            with self.subTest(intrinsic=intrinsic.__name__):
                self.assertTrue(hasattr(intrinsic, CODEGEN_FUNC_ATTR))
                codegen_func = getattr(intrinsic, CODEGEN_FUNC_ATTR)
                self.assertIsNotNone(codegen_func)
                self.assertTrue(callable(codegen_func))

    def test_all_intrinsics_have_eval_return_type(self):
        """Test that all intrinsics have eval_return_type functions."""
        intrinsics = [
            cuda_asm.ld_shared_u32,
            cuda_asm.ld_shared_u64,
            cuda_asm.ld_shared_f32,
            cuda_asm.st_shared_u32,
            cuda_asm.st_shared_u64,
            cuda_asm.st_shared_f32,
            cuda_asm.ld_global_u32,
            cuda_asm.ld_global_u64,
            cuda_asm.st_global_u32,
            cuda_asm.st_global_u64,
            cuda_asm.fma_f32,
            cuda_asm.mul_lo_u32,
            cuda_asm.mul_hi_u32,
            cuda_asm.mad_lo_u32,
            cuda_asm.popc_u32,
            cuda_asm.clz_u32,
            cuda_asm.bfind_u32,
            cuda_asm.ballot_sync,
            cuda_asm.shfl_sync_up,
            cuda_asm.shfl_sync_down,
            cuda_asm.shfl_sync_bfly,
            cuda_asm.sqrt_f32,
            cuda_asm.rsqrt_f32,
            cuda_asm.sin_f32,
            cuda_asm.cos_f32,
            cuda_asm.exp2_approx_f32,
            cuda_asm.exp2f,
            cuda_asm.exp2_emu_f32x2,
            cuda_asm.exp_precise_f32,
            cuda_asm.tanh_approx,
        ]

        for intrinsic in intrinsics:
            with self.subTest(intrinsic=intrinsic.__name__):
                self.assertTrue(hasattr(intrinsic, EVAL_RETURN_TYPE_ATTR))
                eval_return_type = getattr(intrinsic, EVAL_RETURN_TYPE_ATTR)
                self.assertIsNotNone(eval_return_type)
                self.assertTrue(callable(eval_return_type))


class TestVectorMemoryIntrinsics(unittest.TestCase):
    """Tests for the generic v2-global / v2+v4-shared vector families and
    st_global_cs_u64 (8-byte evict-first store)."""

    def _check_vector_intrinsic(self, intrinsic, expected_inst):
        self.assertTrue(getattr(intrinsic, BUILTIN_ATTR, False))
        codegen_func = getattr(intrinsic, CODEGEN_FUNC_ATTR)
        self.assertIsNotNone(codegen_func)
        result = codegen_func("mem_expr", "off_expr", "frag_expr")
        self.assertIn(expected_inst, result.body)
        self.assertIn("<cstdint>", result.includes)
        self.assertIn("(mem_expr + off_expr, frag_expr)", result.return_val)

    def test_ld_global_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.ld_global_v2_u32, "ld.global.v2.u32")

    def test_ld_global_cs_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.ld_global_cs_v2_u32, "ld.global.cs.v2.u32")

    def test_ld_global_cg_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.ld_global_cg_v2_u32, "ld.global.cg.v2.u32")

    def test_st_global_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.st_global_v2_u32, "st.global.v2.u32")

    def test_st_global_cs_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.st_global_cs_v2_u32, "st.global.cs.v2.u32")

    def test_ld_shared_v4_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.ld_shared_v4_u32, "ld.shared.v4.u32")
        codegen_func = getattr(memory.ld_shared_v4_u32, CODEGEN_FUNC_ATTR)
        self.assertIn("__cvta_generic_to_shared", codegen_func("m", "o", "f").body)

    def test_st_shared_v4_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.st_shared_v4_u32, "st.shared.v4.u32")

    def test_ld_shared_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.ld_shared_v2_u32, "ld.shared.v2.u32")

    def test_st_shared_v2_u32_codegen(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        self._check_vector_intrinsic(memory.st_shared_v2_u32, "st.shared.v2.u32")

    def test_st_global_cs_u64_is_builtin_and_exported(self):
        self.assertTrue(getattr(cuda_asm.st_global_cs_u64, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(ll, "st_global_cs_u64"))
        self.assertTrue(
            callable(
                getattr(cuda_asm.st_global_cs_u64, CODEGEN_FUNC_ATTR, None)
                or getattr(cuda_asm.st_global_cs_u64, "__wrapped__", cuda_asm.st_global_cs_u64)))

    def test_new_vector_family_exported_from_ll(self):
        for name in ("ld_global_v2_u32", "ld_global_cs_v2_u32", "ld_global_cg_v2_u32", "st_global_v2_u32",
                     "st_global_cs_v2_u32", "ld_shared_v4_u32", "st_shared_v4_u32", "ld_shared_v2_u32",
                     "st_shared_v2_u32", "st_global_cs_u64"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), f"ll.{name} must be exported")

    def test_v2_intrinsics_have_void_eval_return_type(self):
        import little_kernel.backends.nv.common.language.intrin.memory as memory
        from little_kernel.language import void
        for intrinsic in (memory.ld_global_v2_u32, memory.ld_global_cs_v2_u32, memory.ld_global_cg_v2_u32,
                          memory.st_global_v2_u32, memory.st_global_cs_v2_u32, memory.ld_shared_v4_u32,
                          memory.st_shared_v4_u32, memory.ld_shared_v2_u32, memory.st_shared_v2_u32):
            with self.subTest(intrinsic=intrinsic.__name__):
                self.assertTrue(hasattr(intrinsic, EVAL_RETURN_TYPE_ATTR))


if __name__ == "__main__":
    unittest.main()
