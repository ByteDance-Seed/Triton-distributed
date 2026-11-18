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
"""Bridge-path regression for bare-statement ll.alloc_shared_memory.

``test_alloc_shared_memory_dtype_attr`` locks the direct ``codegen_cuda``
entry point.  The failure this covers only showed up on the *compiled*
path used by frontends (``frontend.lower`` -> ``ai_compiler_eval.lower``
-> ``vibe_cuda_eval.export`` -> PASSES -> renderer -> emitted .cu): a
bare ``ll.alloc_shared_memory(...)`` expression statement in an
``is_entry=True`` kernel must splice the full declaration

    __shared__ [__align__(N)] <ctype> <name>[elems];

for every independent buffer, register each name for later indexing, and
produce CUDA nvcc accepts.  It must never emit a truncated attribute
line (nvcc spells ``__shared__`` as ``__attribute__((shared))`` in its
caret diagnostics, so a declaration cut off after the attribute reports
"expected an expression").  An assignment target around the call is also
legal -- but the declaration must still be spliced as its own statement
(plus a pointer alias when the target name differs from the declared
buffer), never wrapped in a pointer initializer such as
``float* s = __shared__ float s[256];``.

All assertions are CPU-only: the bridge lowers to .cu text and, when a
toolkit is present, ``nvcc -ptx`` syntax-compiles the emitted source
(no GPU/execution required).
"""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

# The exact capability reproducer: one aligned float buffer used through a
# barrier, declared by a bare expression statement.
DRAFT_SINGLE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def smem_decl_repro(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
                    n: ll.int32) -> ll.void:
    ll.alloc_shared_memory("s_buf", ll.float32, 256, 16)
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        s_buf[i % 256] = x[i]
    ll.__syncthreads()
    if i < n:
        y[i] = s_buf[i % 256] * 1.0
'''

# Five independent shared buffers (mixed dtypes/alignments) plus a local
# array: the multi-declaration shape that failed in the entry kernel, and
# the identical check requested for ll.alloc_local_memory.
DRAFT_MULTI = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def smem_multi(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
               n: ll.int32) -> ll.void:
    ll.alloc_shared_memory("sa_sh", ll.float32, 64, 16)
    ll.alloc_shared_memory("sb_sh", ll.float32, 64)
    ll.alloc_shared_memory("sk_sh", ll.int32, 128, 16)
    ll.alloc_shared_memory("sq_sh", ll.int64, 32, 16)
    ll.alloc_shared_memory("sv_sh", ll.float32, 64)
    ll.alloc_local_memory("sl_buf", ll.int32, 64)
    i: ll.int32 = ll.blockIdx_x() * 64 + ll.threadIdx_x()
    if i < n:
        sa_sh[i % 64] = x[i]
        sb_sh[i % 64] = sa_sh[i % 64]
        sk_sh[i % 128] = i
        sq_sh[i % 32] = i
        sv_sh[i % 64] = sb_sh[i % 64]
        sl_buf[i % 64] = i
    ll.__syncthreads()
    if i < n:
        y[i] = (sa_sh[i % 64] + sb_sh[i % 64] + sv_sh[i % 64]) * 1.0
'''

# Assignment-wrapped declaration calls are legal, but codegen must splice
# the declaration as its own statement -- never inline it into a pointer
# initializer (which is not C++).  The same-name target binds the array;
# the distinct target name below binds a pointer alias to it.
DRAFT_ASSIGN = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def smem_assign(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
                n: ll.int32) -> ll.void:
    s_buf = ll.alloc_shared_memory("s_buf", ll.float32, 256, 16)
    tile = ll.alloc_shared_memory("s_tile", ll.float32, 64)
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        s_buf[i % 256] = x[i]
        tile[i % 64] = s_buf[i % 256]
    ll.__syncthreads()
    if i < n:
        y[i] = tile[i % 64]
'''


def _spec(kernel="smem_decl_repro"):
    return {
        "kernel": kernel,
        "function": "lk_forward",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
        "equalities": [],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
        "launch": {"block": [256, 1, 1], "grid": ["ceil_div(x[0], 256)", 1, 1]},
    }


def _lower(root: Path, source: str, kernel: str) -> Path:
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(json.dumps(_spec(kernel)))
    out = root / "output"
    lower(task, program, out, 0)
    return out


def _kernel_cu(out: Path) -> str:
    return next(out.glob("kernels/*.cu")).read_text()


# A complete static shared declaration: optional alignment, then a C
# type, a name and a constant extent -- all on one line.
_SHARED_DECL_RE = re.compile(r'^\s*__shared__(?: __align__\((?P<align>\d+)\))? '
                             r'(?P<ctype>[A-Za-z_]\w*) (?P<name>[A-Za-z_]\w*)'
                             r'\[(?P<elems>\d+)\];\s*$')

# A complete local array declaration: type name[extent]; with no linkage.
_LOCAL_DECL_RE = re.compile(r'^\s*(?P<ctype>[A-Za-z_]\w*) (?P<name>[A-Za-z_]\w*)'
                            r'\[(?P<elems>\d+)\];\s*$')


def _parse_shared_decls(cu: str):
    decls = []
    for line in cu.splitlines():
        if "__shared__" not in line:
            continue
        match = _SHARED_DECL_RE.match(line)
        assert match is not None, ("incomplete/truncated __shared__ declaration emitted by the "
                                   f"bridge path: {line!r}")
        decls.append(match.groupdict())
        # The declared name must be an unquoted C identifier.
        assert '"' not in line, f"quoted buffer name in declaration: {line!r}"
    return decls


class AllocSharedMemoryBareStatementBridgeTests(unittest.TestCase):

    def test_single_buffer_full_declaration_through_bridge(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT_SINGLE, "smem_decl_repro"))
        decls = _parse_shared_decls(cu)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0], {"align": "16", "ctype": "float", "name": "s_buf", "elems": "256"})
        # The name is registered for later indexing (store + reload).
        self.assertIn("s_buf[i % 256] = x[i];", cu)
        self.assertIn("s_buf[i % 256] * 1.0f", cu)
        # nvcc's spelling of a bare attribute must never appear alone.
        self.assertNotIn("__attribute__((shared))\n", cu)
        for line in cu.splitlines():
            self.assertNotEqual(line.strip(), "__shared__;")

    def test_five_buffers_each_get_full_declaration(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT_MULTI, "smem_multi"))
        decls = _parse_shared_decls(cu)
        by_name = {decl["name"]: decl for decl in decls}
        self.assertEqual(set(by_name), {"sa_sh", "sb_sh", "sk_sh", "sq_sh", "sv_sh"})
        self.assertEqual(by_name["sa_sh"], {"align": "16", "ctype": "float", "name": "sa_sh", "elems": "64"})
        self.assertEqual(by_name["sb_sh"], {"align": None, "ctype": "float", "name": "sb_sh", "elems": "64"})
        self.assertEqual(by_name["sk_sh"], {"align": "16", "ctype": "int32_t", "name": "sk_sh", "elems": "128"})
        self.assertEqual(by_name["sq_sh"], {"align": "16", "ctype": "int64_t", "name": "sq_sh", "elems": "32"})
        # The stock local-memory builtin follows the identical convention.
        local = [
            m.groupdict()
            for line in cu.splitlines()
            for m in [_LOCAL_DECL_RE.match(line)]
            if m and m.group("name") == "sl_buf" and "__shared__" not in line
        ]
        self.assertEqual(local, [{"ctype": "int32_t", "name": "sl_buf", "elems": "64"}])
        # Every declared buffer is reachable by later indexing.
        for name in by_name:
            self.assertIn(f"{name}[", cu)

    def test_assignment_wrapped_call_splices_full_declaration(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT_ASSIGN, "smem_assign"))
        decls = _parse_shared_decls(cu)
        by_name = {decl["name"]: decl for decl in decls}
        # Both buffers declared exactly once, fully and unquoted.
        self.assertEqual(set(by_name), {"s_buf", "s_tile"})
        self.assertEqual(by_name["s_buf"], {"align": "16", "ctype": "float", "name": "s_buf", "elems": "256"})
        # A declaration is never inlined into a pointer initializer.
        self.assertNotIn("= __shared__", cu)
        # Same-name target binds the array directly (no self-alias)...
        self.assertNotIn("auto* s_buf =", cu)
        # ...distinct target name gets one pointer alias to its buffer.
        self.assertIn("auto* tile = s_tile;", cu)
        # Both names are reachable by later indexing (store + reload).
        self.assertIn("s_buf[i % 256] = x[i];", cu)
        self.assertIn("tile[i % 64] = s_buf", cu)
        self.assertIn("y[i] = tile[i % 64];", cu)
        # nvcc's spelling of a bare attribute must never appear alone.
        self.assertNotIn("__attribute__((shared))\n", cu)

    @unittest.skipIf(shutil.which("nvcc") is None, "nvcc not available: compile-only check skipped")
    def test_emitted_cu_compiles_with_nvcc(self):
        # Syntax compile to PTX: no device or execution needed. This is
        # the exact gate the reported failure broke before any kernel
        # semantics were involved.
        with tempfile.TemporaryDirectory() as d:
            out = _lower(Path(d), DRAFT_MULTI, "smem_multi")
            cu = next(out.glob("kernels/*.cu"))
            ptx = Path(d) / "out.ptx"
            proc = subprocess.run(
                [shutil.which("nvcc"), "-ptx", str(cu), "-o", str(ptx)], capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, f"nvcc rejected the emitted .cu:\n{proc.stderr}")


if __name__ == "__main__":
    unittest.main()
