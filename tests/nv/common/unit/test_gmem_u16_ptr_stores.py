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
"""void*-keyed 16-bit GMEM accesses: gmem_ld_u16 / gmem_st_u16.

Render-layer legality tests (no GPU), completing the byte-addressed
scalar move family at the 2-byte granule.  The pinned API offered u8 and
u32 void*-keyed scalar twins, but no typed u16 op, so an aligned 2-byte
lane (e.g. one packed bf16x2->fp8x2 rope pair per thread lane, or the
2-byte granule of a ragged tail wider vector stores cannot touch) could
only be spelled as two adjacent gmem_st_u8/gmem_ld_u8 accesses despite
the address being 2-byte aligned.

  * gmem_ld_u16(ptr) zero-extends the 16-bit value into a uint32
    register (ld.global.u16);
  * gmem_st_u16(ptr, v) stores the low 16 bits of v (st.global.u16);
  * a gmem_ld_u16 -> gmem_st_u16 pair is a bit-exact two-byte move with
    NO numeric conversion (no cvt asm, no fp8/half conversion headers);
  * legality precondition is frontend-side: the byte address must be
    2-byte aligned and in bounds, identical to dereferencing a uint16_t*
    in CUDA C; the compiler deliberately does not infer or promote
    alignment.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def u16_move(src: ll.Tensor[ll.uint8], dst: ll.Tensor[ll.uint8],
             n: ll.int32) -> ll.void:
    # Generic two-byte-keyed move: one thread per aligned 2-byte lane,
    # addressed byte-keyed through ptr_byte_offset so the same kernel
    # shape serves any opaque payload dtype.
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        v: ll.uint32 = ll.gmem_ld_u16(ll.ptr_byte_offset(src, i * 2))
        ll.gmem_st_u16(ll.ptr_byte_offset(dst, i * 2), v)
'''


def _spec():
    return {
        "kernel": "u16_move",
        "function": "u16_move",
        "inputs": [
            {"name": "src", "dtype": "uint8", "rank": 1},
        ],
        "outputs": [
            {"name": "dst", "dtype": "uint8", "shape": ["src[0]"]},
        ],
        "scalars": [{"name": "n", "expr": "src[0]"}],
        "kernel_args": ["src", "dst", "n"],
        "forward_args": ["src"],
        "launch": {"grid": ["ceil_div(n, 128)", 1, 1], "block": [128, 1, 1]},
    }


class GmemU16PtrStoreTests(unittest.TestCase):

    def _render(self, directory, source=SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "u16_move")
        return _render_cuda(spec, kernel)

    def test_u16_ptr_accesses_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Typed 16-bit load and store via uint16_t pointer casts: ld/st
        # .global.u16 semantics with the low 16 bits of the uint32 value.
        self.assertIn("(uint32_t)(*reinterpret_cast<const uint16_t*>", cuda)
        self.assertIn("*reinterpret_cast<uint16_t*>", cuda)
        self.assertIn("= (uint16_t)(", cuda)

    def test_no_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Pure bit moves: no cvt asm and no fp16/bf16/fp8 conversion
        # headers — bits in == bits out.
        self.assertNotIn("cvt.rn", cuda)
        self.assertNotIn("<cuda_fp8.h>", cuda)
        self.assertNotIn("<cuda_fp16.h>", cuda)
        self.assertNotIn("<cuda_bf16.h>", cuda)
        self.assertNotIn("__nv_cvt_", cuda)

    def test_low_16_bits_truncation_contract(self):
        # The capability semantics: takes a uint32 value and stores its
        # low 16 bits.  Feed a value with high bits set; the store must
        # still narrow via (uint16_t) and must NOT fall back to two
        # adjacent u8 stores.
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def u16_move(out: ll.Tensor[ll.uint8], off: ll.int32) -> ll.void:
    addr = ll.ptr_byte_offset(out, ll.int64(off) * 576 + 512)
    ll.gmem_st_u16(addr, ll.uint32(7))
'''
        # Dedicated spec matching this kernel's (out, off) signature.
        spec = {
            "kernel": "u16_move",
            "function": "u16_move",
            "inputs": [
                {"name": "off_in", "dtype": "int32", "rank": 1},
            ],
            "outputs": [
                {"name": "out", "dtype": "uint8", "shape": ["off_in[0]"]},
            ],
            "scalars": [{"name": "off", "expr": "off_in[0]"}],
            "kernel_args": ["out", "off"],
            "forward_args": ["off_in"],
            "launch": {"grid": [1, 1, 1], "block": [1, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "export.json").write_text(json.dumps(spec))
            kernel_path = root / "kernel.py"
            kernel_path.write_text(source)
            loaded_spec = _load_spec(root / "export.json")
            kernel = _load_kernel(kernel_path, "u16_move")
            cuda = _render_cuda(loaded_spec, kernel)
        self.assertIn("*reinterpret_cast<uint16_t*>", cuda)
        self.assertIn("= (uint16_t)(", cuda)
        # One aligned u16 store, not the two adjacent u8-store spelling.
        self.assertNotIn("*reinterpret_cast<uint8_t*>", cuda)

    def test_no_u16_ops_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def u16_move(src: ll.Tensor[ll.uint8], dst: ll.Tensor[ll.uint8],
             n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        dst[i] = src[i]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, source)
        self.assertNotIn("uint16_t*", cuda)


if __name__ == "__main__":
    unittest.main()
