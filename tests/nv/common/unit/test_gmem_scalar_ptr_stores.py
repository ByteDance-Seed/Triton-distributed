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
"""void*-keyed scalar GMEM stores: gmem_st_u8 / gmem_st_u32 / _cg / _cs twins.

Render-layer legality tests (no GPU), completing the byte-addressed scalar
move family whose LOAD half already shipped: gmem_ld_u8 and the
gmem_ld_{nc,cs,cg}_u32 loads compose with ptr_byte_offset for
dtype-agnostic gathered moves, but there was NO void*-keyed scalar store
twin to write a loaded byte/word back.  Ragged tails of bit-exact moves of
low-precision payloads (e.g. the sub-16-byte tail of a gathered fp8 row
with H not a multiple of 16, or the tail words of a gathered int32
scale-factor row with P not a multiple of 4) are exactly the cases where
the 16-byte vector twins' alignment contract does not hold, so the only
fallbacks were the typed element-indexed gmem_st_scalar (unavailable when
addressing is byte-keyed) or a wider vector store (over-writes).

  * gmem_st_u8(ptr, v) stores the low 8 bits of v (st.global.u8); a
    gmem_ld_u8 -> gmem_st_u8 pair is a bit-exact one-byte move with no
    alignment requirement beyond the byte itself;
  * gmem_st_u32(ptr, v) stores the raw uint32 bits (st.global.u32); a
    gmem_ld_*_u32 -> gmem_st_u32 pair is a bit-exact 4-byte move;
  * gmem_st_cg_u32 / gmem_st_cs_u32 move identical bits with the L2-only
    (__stcg) and evict-first (__stcs) cache policies, mirroring the load
    policy set (PTX has no st.nc, so no _nc store twin);
  * no numeric conversion or cvt/header appears anywhere: bits in ==
    bits out.
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
def tail_move(src_b: ll.Tensor[ll.uint8], dst_b: ll.Tensor[ll.uint8],
              src_w: ll.Tensor[ll.int32], dst_w: ll.Tensor[ll.int32],
              ntail: ll.int32) -> ll.void:
    # Generic ragged-tail move: sub-16-byte byte tail plus sub-16-byte
    # int32 word tail, addressed byte-keyed through ptr_byte_offset so the
    # same kernel shape serves any opaque payload dtype.
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < ntail:
        b: ll.uint32 = ll.gmem_ld_u8(ll.ptr_byte_offset(src_b, i))
        w: ll.uint32 = ll.gmem_ld_nc_u32(ll.ptr_byte_offset(src_w, i * 4))
        w2: ll.uint32 = ll.gmem_ld_cs_u32(ll.ptr_byte_offset(src_w, i * 4))
        w3: ll.uint32 = ll.gmem_ld_cg_u32(ll.ptr_byte_offset(src_w, i * 4))
        ll.gmem_st_u8(ll.ptr_byte_offset(dst_b, i), b)
        ll.gmem_st_u32(ll.ptr_byte_offset(dst_w, i * 4), w)
        ll.gmem_st_cg_u32(ll.ptr_byte_offset(dst_w, i * 4), w2)
        ll.gmem_st_cs_u32(ll.ptr_byte_offset(dst_w, i * 4), w3)
'''


def _spec():
    return {
        "kernel":
        "tail_move",
        "function":
        "tail_move",
        "inputs": [
            {"name": "src_b", "dtype": "uint8", "rank": 1},
            {"name": "src_w", "dtype": "int32", "rank": 1},
        ],
        "outputs": [
            {"name": "dst_b", "dtype": "uint8", "shape": ["src_b[0]"]},
            {"name": "dst_w", "dtype": "int32", "shape": ["src_w[0]"]},
        ],
        "scalars": [{"name": "ntail", "expr": "src_b[0]"}],
        "kernel_args": ["src_b", "dst_b", "src_w", "dst_w", "ntail"],
        "forward_args": ["src_b", "src_w"],
        "launch": {"grid": ["ceil_div(ntail, 128)", 1, 1], "block": [128, 1, 1]},
    }


class GmemScalarPtrStoreTests(unittest.TestCase):

    def _render(self, directory, source=SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "tail_move")
        return _render_cuda(spec, kernel)

    def test_scalar_ptr_stores_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Plain byte store: no alignment precondition, no widening.
        self.assertIn("*reinterpret_cast<uint8_t*>", cuda)
        # Plain word store plus the __stcg/__stcs policy twins.
        self.assertIn("*reinterpret_cast<uint32_t*>", cuda)
        self.assertIn("__stcg(reinterpret_cast<uint32_t*>", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint32_t*>", cuda)
        # Load twins still render on the read side.
        self.assertIn("__ldg(reinterpret_cast<const uint32_t*>", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint32_t*>", cuda)
        self.assertIn("__ldcg(reinterpret_cast<const uint32_t*>", cuda)

    def test_no_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Pure bit moves: no cvt asm and no fp8/half conversion headers.
        self.assertNotIn("cvt.rn", cuda)
        self.assertNotIn("<cuda_fp8.h>", cuda)
        self.assertNotIn("__nv_cvt_", cuda)

    def test_no_stores_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def tail_move(src_b: ll.Tensor[ll.uint8], dst_b: ll.Tensor[ll.uint8],
              src_w: ll.Tensor[ll.int32], dst_w: ll.Tensor[ll.int32],
              ntail: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < ntail:
        dst_b[i] = src_b[i]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, source)
        self.assertNotIn("__stcg(", cuda)
        self.assertNotIn("__stcs(", cuda)

    def test_fp8_gather_tail_canary(self):
        # Canary for the audited operation shape: a bit-exact index-table
        # gather of fp8 payload rows (uint8 storage, arbitrary H with a
        # sub-16-byte tail) plus int32 scale-factor rows (arbitrary P, word
        # tail).  16-byte bodies go through the u8x16/v4u32 vector twins;
        # ragged tails must now render through the scalar ptr stores
        # instead of falling off the family.  Render layer only: exact
        # address arithmetic stays a frontend concern.
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def tail_move(storage: ll.Tensor[ll.uint8], sf_table: ll.Tensor[ll.int32],
              idx: ll.Tensor[ll.int32], data: ll.Tensor[ll.uint8],
              sf: ll.Tensor[ll.int32], H: ll.int32, P: ll.int32) -> ll.void:
    # One thread per gathered output row: copy H payload bytes then P
    # scale words from the gathered source row, both on scalar paths.
    r: ll.int32 = ll.blockIdx_x() * 64 + ll.threadIdx_x()
    if r < 4096:
        src: ll.int32 = idx[r]
        c: ll.int32 = 0
        while c < H:
            b: ll.uint32 = ll.gmem_ld_u8(ll.ptr_byte_offset(storage, src * H + c))
            ll.gmem_st_u8(ll.ptr_byte_offset(data, r * H + c), b)
            c = c + 1
        p: ll.int32 = 0
        while p < P:
            w: ll.uint32 = ll.gmem_ld_nc_u32(ll.ptr_byte_offset(sf_table, (src * P + p) * 4))
            ll.gmem_st_cs_u32(ll.ptr_byte_offset(sf, (r * P + p) * 4), w)
            p = p + 1
'''
        root_spec = {
            "kernel":
            "tail_move",
            "function":
            "tail_move",
            "inputs": [
                {"name": "storage", "dtype": "float8_e4m3fn", "rank": 1},
                {"name": "sf_table", "dtype": "int32", "rank": 1},
                {"name": "idx", "dtype": "int32", "rank": 1},
            ],
            "outputs": [
                {"name": "data", "dtype": "float8_e4m3fn", "shape": ["idx[0]"]},
                {"name": "sf", "dtype": "int32", "shape": ["idx[0]"]},
            ],
            "scalars": [{"name": "H", "dtype": "int32", "expr": "7"}, {"name": "P", "dtype": "int32", "expr": "3"}],
            "kernel_args": ["storage", "sf_table", "idx", "data", "sf", "H", "P"],
            "forward_args": ["storage", "sf_table", "idx"],
            "launch": {"grid": ["ceil_div(idx[0], 64)", 1, 1], "block": [64, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "export.json").write_text(json.dumps(root_spec))
            kernel_path = root / "kernel.py"
            kernel_path.write_text(source)
            spec = _load_spec(root / "export.json")
            kernel = _load_kernel(kernel_path, "tail_move")
            cuda = _render_cuda(spec, kernel)
        # fp8 payload storage arrives byte-addressed (uint8_t* at the ABI);
        # the scalar byte/word tail stores and the read-only sf load render.
        self.assertIn("*reinterpret_cast<uint8_t*>", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint32_t*>", cuda)
        self.assertIn("__ldg(reinterpret_cast<const uint32_t*>", cuda)
        self.assertNotIn("__nv_cvt_", cuda)
        self.assertNotIn("cvt.rn", cuda)


if __name__ == "__main__":
    unittest.main()
