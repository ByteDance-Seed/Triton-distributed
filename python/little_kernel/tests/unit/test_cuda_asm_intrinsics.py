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
from little_kernel.language.intrin import cuda_asm
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
        ]

        for intrinsic in intrinsics:
            with self.subTest(intrinsic=intrinsic.__name__):
                self.assertTrue(hasattr(intrinsic, EVAL_RETURN_TYPE_ATTR))
                eval_return_type = getattr(intrinsic, EVAL_RETURN_TYPE_ATTR)
                self.assertIsNotNone(eval_return_type)
                self.assertTrue(callable(eval_return_type))


if __name__ == "__main__":
    unittest.main()
