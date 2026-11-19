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
"""Guard full unrolling against exiting edges and large compile-time ranges."""

import ast
import shutil
import subprocess
import tracemalloc

import pytest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.codegen.visitors.control_flow_codegen import ControlFlowCodegenMixin


@pytest.mark.parametrize("statement", ["break", "continue"])
def test_direct_iteration_exit_keeps_a_real_loop(statement):
    loop = ast.parse(f"for i in range(4):\n    {statement}\n").body[0]
    assert ControlFlowCodegenMixin()._static_unroll_values(loop) is None


@pytest.mark.parametrize("stop", [1_000_000, 1 << 64])
def test_large_static_range_is_bounded_without_materializing_iterations(stop):
    loop = ast.parse(f"for i in range({stop}):\n    pass\n").body[0]
    tracemalloc.start()
    try:
        result = ControlFlowCodegenMixin()._static_unroll_values(loop)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result is None
    # Enumerating a million Python ints costs tens of MB. The eligibility
    # check should use bounded memory, independent of the source trip count.
    assert peak < 1024 * 1024


def test_exit_from_nested_loop_does_not_block_outer_unroll():
    loop = ast.parse("for i in range(4):\n    for j in range(2):\n        break\n").body[0]
    assert ControlFlowCodegenMixin()._static_unroll_values(loop) == [0, 1, 2, 3]


@pytest.mark.parametrize("statement", ["break", "continue"])
def test_unroll_with_direct_iteration_exit_compiles(statement, tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not available")
    source = ("def k(out: ll.Tensor[ll.int32]) -> ll.void:\n"
              "    for i in ll.unroll(range(4)):\n"
              f"        {statement}\n"
              "    out[0] = 1\n")
    cuda = codegen_cuda(ast.parse(source), ctx={"ll": ll})
    path = tmp_path / "kernel.cu"
    path.write_text(cuda)
    result = subprocess.run(
        [nvcc, "-std=c++17", "-arch=sm_90a", "-c",
         str(path), "-o", str(tmp_path / "kernel.o")], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
