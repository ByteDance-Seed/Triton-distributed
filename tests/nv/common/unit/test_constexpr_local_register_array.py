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
"""Constexpr-sized local register arrays (semantic key
``ir.local_register_array.constexpr_sized``).

A constexpr constructor scalar is baked to a literal before tracing, so it
is a trace-time integer: arithmetic over it sizes a local C array and bounds
a fully unrolled loop, and one parameterized IR program specializes the
register-resident row tile per launch (CPT=2 -> ``T raw[8]`` and two unrolled
copies, CPT=4 -> ``T raw[16]`` and four), while the constexpr name leaves the
runtime kernel ABI.

The declaration allocators (``alloc_local_memory`` etc.) are pass-inserted
as bare statements, but the direct assignment spelling
(``raw = ll.alloc_local_memory("raw", ...)``) must emit the declaration
itself rather than the ill-formed ``T* raw = T raw[N];`` it produced before.
"""

import ast
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
import little_kernel as lk
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_tile(a: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32],
             n: ll.int32, cpt: ll.int32) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    raw = ll.alloc_local_memory("raw", ll.int32, 4 * cpt)
    acc: ll.int32 = 0
    for c in ll.unroll(range(cpt)):
        idx: ll.int32 = tid * 4 * cpt + c
        if idx < n:
            raw[c] = a[idx]
            acc += raw[c]
    out[tid] = acc
'''

EMPTY_SOURCE = SOURCE.replace('    raw = ll.alloc_local_memory("raw", ll.int32, 4 * cpt)\n',
                              '    raw: ll.Tensor[ll.int32] = ll.empty([4 * cpt], dtype=ll.int32, '
                              'scope="local")\n')


def _spec(cpt):
    return {
        "kernel": "row_tile",
        "function": "tile",
        "inputs": [{"name": "a", "dtype": "int32", "rank": 1}],
        "output": {"name": "out", "dtype": "int32", "shape": ["256"]},
        "constructor": [{"name": "cpt", "dtype": "int32", "constexpr": cpt}],
        "scalars": [{"name": "n", "expr": "a[0]"}],
        "kernel_args": ["a", "out", "n", "cpt"],
        "launch": {"grid": [1, 1, 1], "block": [256, 1, 1]},
    }


def _lower(source, cpt):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task = root / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = root / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        (program / "export.json").write_text(json.dumps(_spec(cpt)))
        output = root / "out"
        lower(task, program, output, 0)
        return (output / "kernels/agent_lk_tile.cu").read_text()


def _emit(body):
    source = ("def k(a: ll.Tensor[ll.int32]) -> ll.void:\n" + body)
    return codegen_cuda(ast.parse(source), ctx={"ll": ll}, emit_header=False)


class ConstexprLocalRegisterArrayTests(unittest.TestCase):

    def test_recompiling_same_kernel_refreshes_specialized_array_shapes(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def specialize(out: ll.Tensor[ll.int32], width: ll.int32) -> ll.void:
            raw = ll.empty([2, width], dtype=ll.int32, scope="local")
            raw[1, width - 1] = 1
            out[0] = raw[1, width - 1]

        for width in (2, 4, 2):
            cuda = specialize.compile(PASSES["cuda"], codegen_cuda, num_threads=32, constexpr_args={"width": width})
            self.assertIn(f"int32_t raw[{2 * width}];", cuda)
            self.assertIn(f"raw[(1) * {width} + ({width - 1})]", cuda)

    def test_extent_and_unroll_specialize_per_launch(self):
        for cpt in (2, 4):
            with self.subTest(cpt=cpt):
                cuda = _lower(SOURCE, cpt)
                # The extent is the trace-time product 4 * <baked cpt>.
                self.assertIn(f"int32_t raw[{4 * cpt}];", cuda)
                self.assertNotIn("raw = int32_t raw", cuda)
                # The unroll trip count is the same baked literal.
                self.assertEqual(cuda.count("acc += raw["), cpt)
                self.assertIn(f"tid * 4 * {cpt}", cuda)
                # The constexpr constructor scalar leaves the runtime ABI.
                entry = cuda[cuda.index("void agent_row_tile"):cuda.index("{", cuda.index("void agent_row_tile"))]
                self.assertNotIn("cpt", entry)
                self.assertIn("int32_t n", entry)

    def test_public_empty_path_specializes_identically(self):
        for cpt in (2, 4):
            with self.subTest(cpt=cpt):
                cuda = _lower(EMPTY_SOURCE, cpt)
                self.assertIn(f"int32_t raw[{4 * cpt}];", cuda)
                self.assertEqual(cuda.count("acc += raw["), cpt)

    def test_assignment_spelling_emits_plain_declaration(self):
        cuda = _emit('    raw = ll.alloc_local_memory("raw", ll.int32, 8)\n'
                     '    raw[0] = a[0]\n')
        self.assertIn("int32_t raw[8];", cuda)
        self.assertNotIn("int32_t* raw", cuda)

    def test_annotated_assignment_spelling_emits_plain_declaration(self):
        cuda = _emit('    raw: ll.Tensor[ll.int32] = ll.alloc_local_memory('
                     '"raw", ll.int32, 8)\n'
                     '    raw[0] = a[0]\n')
        self.assertIn("int32_t raw[8];", cuda)
        self.assertNotIn("int32_t* raw", cuda)

    def test_shared_declaration_assignment_is_well_formed(self):
        cuda = _emit('    s = ll.alloc_shared_memory("s", ll.int32, 4)\n'
                     '    s[0] = a[0]\n')
        self.assertIn("__shared__ int32_t s[4];", cuda)
        self.assertNotIn("int32_t* s", cuda)

    def test_mismatched_declaration_name_is_a_compile_error(self):
        with self.assertRaisesRegex(ValueError, "declares array 'y'.*assigned to 'x'"):
            _emit('    x = ll.alloc_local_memory("y", ll.int32, 8)\n'
                  '    x[0] = a[0]\n')

    def test_annotated_mismatched_declaration_name_is_a_compile_error(self):
        with self.assertRaisesRegex(ValueError, "declares array 'y'.*assigned to 'x'"):
            _emit('    x: ll.Tensor[ll.int32] = ll.alloc_local_memory('
                  '"y", ll.int32, 8)\n'
                  '    x[0] = a[0]\n')

    @unittest.skipUnless(shutil.which("nvcc"), "nvcc not available")
    def test_generated_specializations_compile(self):
        # Compile-only static check: no GPU device is required.
        for cpt in (2, 4):
            cuda = _lower(SOURCE, cpt)
            with tempfile.TemporaryDirectory() as directory:
                source_path = Path(directory) / "tile.cu"
                source_path.write_text(cuda)
                result = subprocess.run([
                    "nvcc", "-std=c++17", "-arch=sm_90a", "-c",
                    str(source_path), "-o",
                    str(Path(directory) / "tile.o")
                ], capture_output=True, text=True, timeout=300)
                self.assertEqual(result.returncode, 0, f"nvcc rejected CPT={cpt} specialization:\n"
                                 f"{result.stderr}")


if __name__ == "__main__":
    unittest.main()
