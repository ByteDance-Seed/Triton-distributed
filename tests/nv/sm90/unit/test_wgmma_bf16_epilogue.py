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
"""BF16 accumulator stores specialize safely on rendered literal operands."""
import pytest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.sm90.language.intrin.wgmma import codegen_store_acc_to_smem_bf16
from little_kernel.core.passes import PASSES


def test_different_accumulator_widths_can_share_one_translation_unit():

    @lk.ll_kernel(backend="cuda", is_entry=True)
    def epilogue() -> ll.void:
        smem = ll.empty([64 * 256], dtype=ll.bfloat16, scope="shared")
        small = ll.empty([32], dtype=ll.float32, scope="local")
        large = ll.empty([128], dtype=ll.float32, scope="local")
        tid: ll.int32 = ll.threadIdx_x()
        ll.store_acc_to_smem_bf16(smem, small, tid, 0, 32, 256)
        ll.store_acc_to_smem_bf16(smem, large, tid, 0, 128, 256)

    source = epilogue.compile(PASSES["cuda"], codegen_cuda)
    assert "void store_acc_smem_bf16_32_256_fn(" in source
    assert "void store_acc_smem_bf16_128_256_fn(" in source
    assert "i < 8;" in source and "i < 32;" in source


@pytest.mark.parametrize("registers,stride", [("dynamic", "256"), ("128", "stride"), ("64", "64"), ("3", "256")])
def test_epilogue_rejects_nonconstant_or_undersized_layout(registers, stride):
    with pytest.raises(ValueError):
        codegen_store_acc_to_smem_bf16("smem", "acc", "tid", "0", registers, stride)
