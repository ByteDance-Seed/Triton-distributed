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
"""GPU regression tests for the (tensor, elem_offset) global-access intrinsic
families in `little_kernel/language/intrin/memory.py` and the element-index
atomic family in `little_kernel/language/intrin/atomic.py`.

Covers the semantics frontends rely on (see /exchange requests, cycles 2-7):

* ld_global_{,cs,cg_}v4_u32 / st_global_{,cs_}v4_u32 round-trip values
  unchanged -- cache-policy hints never change data (bit-level identity is
  the documented legality contract of the .cs/.cg policies);
* atomic_add_global_{i32,u32,u64,f32} / atomic_cas_global_{i32,u32} perform
  relaxed per-element RMW at (tensor + index) with exact totals, so
  frontends can build persistent-kernel counters/flags without forming
  `&tensor[i]` in kernel source (paired with ll.threadfence for
  producer/consumer ordering);
* st_global_cs_bf16_f32_rn equals torch's f32 -> bf16 RNE cast bitwise and
  its generated CUDA contains `st.global.cs.u16`;
* prefetch_global_l1 / prefetch_global_l2 are numerically inert and emit
  `prefetch.global.L1` / `prefetch.global.L2`.

Legality respected by construction: every vector access is 16 B aligned and
in bounds; prefetch offsets stay inside the source tensor.

Skill consulted: cw.codegen.tilelang-ptx-inline-assembly, wiki commit
2e523d63b315bedc2adb022b750d9af0f42f7428 (E002/E007 covers the out-array
helper-function idiom and constraint-letter/dtype mapping these intrinsics
use).
"""

import pytest

# This module is GPU-only; without torch there is nothing CPU-executable here.
# ``pytest.importorskip`` skips collection rather than erroring; every test is
# additionally gated by ``requires_cuda`` so no GPU work runs without a device.
torch = pytest.importorskip("torch")

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from tests.nv.conftest import requires_cuda

N = 256  # uint32 elements exercised by the vector memory tests
M = 96  # float32 elements exercised by the scalar-store/prefetch test


@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_mem_kernel(src: ll.Tensor[ll.uint32], dst: ll.Tensor[ll.uint32], dst_cs: ll.Tensor[ll.uint32],
                   dst_cg: ll.Tensor[ll.uint32], dst_nc: ll.Tensor[ll.uint32],
                   dst_nc2: ll.Tensor[ll.uint32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    off: ll.int32 = t * 4
    frag = ll.empty([4], dtype=ll.uint32, scope="local")

    # default-policy 128-bit round trip
    ll.ld_global_v4_u32(src, off, frag)
    ll.st_global_v4_u32(dst, off, frag)

    # evict-first streaming round trip
    ll.ld_global_cs_v4_u32(src, off, frag)
    ll.st_global_cs_v4_u32(dst_cs, off, frag)

    # L2-only load (.cg) + default store
    ll.ld_global_cg_v4_u32(src, off, frag)
    ll.st_global_v4_u32(dst_cg, off, frag)

    # read-only (.nc) 128-bit round trip; src is never written by this kernel,
    # so the non-coherent path is legal here
    ll.ld_global_nc_v4_u32(src, off, frag)
    ll.st_global_v4_u32(dst_nc, off, frag)

    # read-only (.nc) 64-bit pair round trip over the first half of src
    off2: ll.int32 = t * 2
    frag2 = ll.empty([2], dtype=ll.uint32, scope="local")
    ll.ld_global_nc_v2_u32(src, off2, frag2)
    dst_nc2[off2] = frag2[0]
    dst_nc2[off2 + 1] = frag2[1]


@lk.ll_kernel(backend="cuda", is_entry=True)
def atomic_kernel(ctr: ll.Tensor[ll.int32], uc32: ll.Tensor[ll.uint32], uctr: ll.Tensor[ll.uint64],
                  fctr: ll.Tensor[ll.float32], flags: ll.Tensor[ll.int32], flags_u: ll.Tensor[ll.uint32],
                  winners: ll.Tensor[ll.int32], winners_u: ll.Tensor[ll.uint32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    g: ll.int32 = ll.blockIdx_x() * 256 + t
    old: ll.int32 = ll.atomic_add_global_i32(ctr, g % 8, 3)  # noqa: F841
    oldu: ll.uint32 = ll.atomic_add_global_u32(uc32, g % 2, 7)  # noqa: F841
    old64: ll.uint64 = ll.atomic_add_global_u64(uctr, g % 4, 5)  # noqa: F841
    oldf: ll.float32 = ll.atomic_add_global_f32(fctr, g % 4, 0.5)  # noqa: F841
    won: ll.int32 = ll.atomic_cas_global_i32(flags, g % 8, 0, 1)
    if won == 0:
        ll.atomic_add_global_i32(winners, g % 8, 1)
    wonu: ll.uint32 = ll.atomic_cas_global_u32(flags_u, g % 8, 0, 1)
    if wonu == 0:
        ll.atomic_add_global_u32(winners_u, g % 8, 1)
    ll.threadfence()


@lk.ll_kernel(backend="cuda", is_entry=True)
def scalar_store_kernel(src: ll.Tensor[ll.float32], dst: ll.Tensor[ll.bfloat16]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = src[t] * 2.0
    ll.prefetch_global_l1(src, (t + 32) % 96)
    ll.prefetch_global_l2(src, t)
    ll.st_global_cs_bf16_f32_rn(dst, t, v)


@lk.ll_kernel(backend="cuda", is_entry=True)
def exch_kernel(flags: ll.Tensor[ll.int32], first_old: ll.Tensor[ll.int32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    g: ll.int32 = ll.blockIdx_x() * 256 + t
    old: ll.int32 = ll.atomic_exch_global_i32(flags, g % 8, 7)
    if old == 0:
        ll.atomic_add_global_i32(first_old, g % 8, 1)


@lk.ll_kernel(backend="cuda", is_entry=True)
def seg_shfl_kernel(out: ll.Tensor[ll.int32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    v: ll.int32 = t
    # segmented down-shuffle, 16 lanes/segment: lane i reads i+1 within its
    # segment; the last lane of each segment reads its own value back
    r: ll.int32 = ll.shfl_down_sync_width(0xFFFFFFFF, v, 1, 16)
    out[t] = r


@requires_cuda
def test_vector_ldst_policies():
    torch.manual_seed(0)
    src = torch.randint(-2**31, 2**31 - 1, (N, ), dtype=torch.int32, device="cuda")
    dst = torch.zeros_like(src)
    dst_cs = torch.zeros_like(src)
    dst_cg = torch.zeros_like(src)
    dst_nc = torch.zeros_like(src)
    dst_nc2 = torch.zeros_like(src)
    kern = vec_mem_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(64, 1, 1))
    kern(src, dst, dst_cs, dst_cg, dst_nc, dst_nc2)
    kern.synchronize()

    assert torch.equal(dst, src), "default v4 round trip mismatch"
    assert torch.equal(dst_cs, src), ".cs v4 round trip mismatch"
    assert torch.equal(dst_cg, src), ".cg v4 load round trip mismatch"
    assert torch.equal(dst_nc, src), ".nc v4 load round trip mismatch"
    assert torch.equal(dst_nc2[:N // 2], src[:N // 2]), \
        ".nc v2 load round trip mismatch"

    # source-level legality check: exact PTX the family documents
    cuda_src = vec_mem_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=64)
    assert "ld.global.v4.u32" in cuda_src
    assert "ld.global.cs.v4.u32" in cuda_src
    assert "ld.global.cg.v4.u32" in cuda_src
    assert "ld.global.nc.v4.u32" in cuda_src
    assert "ld.global.nc.v2.u32" in cuda_src


@requires_cuda
def test_element_atomics():
    ctr = torch.zeros(8, dtype=torch.int32, device="cuda")
    uc32 = torch.zeros(2, dtype=torch.int32, device="cuda")  # u32 bits as i32
    uctr = torch.zeros(4, dtype=torch.uint64, device="cuda")
    fctr = torch.zeros(4, dtype=torch.float32, device="cuda")
    flags = torch.zeros(8, dtype=torch.int32, device="cuda")
    flags_u = torch.zeros(8, dtype=torch.int32, device="cuda")
    winners = torch.zeros(8, dtype=torch.int32, device="cuda")
    winners_u = torch.zeros(8, dtype=torch.int32, device="cuda")
    kern = atomic_kernel.build(PASSES["cuda"], codegen_cuda, grid=(2, 1, 1), block=(256, 1, 1))
    kern(ctr, uc32, uctr, fctr, flags, flags_u, winners, winners_u)
    kern.synchronize()

    # 512 threads; each residue class of g % r is hit 512/r times
    assert torch.equal(ctr, torch.full_like(ctr, 192)), ctr
    assert torch.equal(uc32, torch.full_like(uc32, 1792)), uc32
    assert torch.equal(uctr, torch.full_like(uctr, 128 * 5)), uctr
    assert torch.equal(fctr, torch.full_like(fctr, 64.0)), fctr
    assert torch.equal(winners, torch.ones_like(winners)), winners
    assert torch.equal(winners_u, torch.ones_like(winners_u)), winners_u


@requires_cuda
def test_scalar_cs_bf16_store_and_prefetch():
    torch.manual_seed(2)
    src = torch.empty(M, dtype=torch.float32).uniform_(-10.0, 10.0)
    # edge set in bf16-normal range: +-0, small/large normals, exact values
    src[0:6] = torch.tensor([0.0, -0.0, 2.0**-126, 2.0**-120, 3.0**38, -1.0])
    src = src.cuda()
    dst = torch.zeros(M, dtype=torch.bfloat16, device="cuda")
    kern = scalar_store_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(96, 1, 1))
    kern(src, dst)
    kern.synchronize()

    ref = (src * 2.0).cpu().to(torch.bfloat16)
    got = dst.cpu()
    assert torch.equal(got.view(torch.uint16),
                       ref.view(torch.uint16)), ("st_global_cs_bf16_f32_rn != f32 -> bf16 RNE cast")

    # source-level legality check: the exact PTX the family documents
    cuda_src = scalar_store_kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=96)
    assert "st.global.cs.u16" in cuda_src, "missing st.global.cs.u16 in CUDA"
    assert "prefetch.global.L1" in cuda_src, "missing prefetch.global.L1"
    assert "prefetch.global.L2" in cuda_src, "missing prefetch.global.L2"


@lk.ll_kernel(backend="cuda", is_entry=True)
def v2_store_kernel(vals: ll.Tensor[ll.uint32], dst: ll.Tensor[ll.uint32], dst_cs: ll.Tensor[ll.uint32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    frag = ll.empty([2], dtype=ll.uint32, scope="local")
    frag[0] = vals[t * 2]
    frag[1] = vals[t * 2 + 1]
    ll.st_global_v2_u32(dst, t * 2, frag)
    ll.st_global_cs_v2_u32(dst_cs, t * 2, frag)


@lk.ll_kernel(backend="cuda", is_entry=True)
def quant_kernel(src: ll.Tensor[ll.float32], packed: ll.Tensor[ll.uint32], byt: ll.Tensor[ll.uint8]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    lo: ll.float32 = src[t * 2]
    hi: ll.float32 = src[t * 2 + 1]
    packed[t] = ll.cvt_rn_satfinite_e4m3x2_f32(lo, hi)
    byt[t * 2] = ll.cvt_rn_satfinite_e4m3_f32(lo)
    byt[t * 2 + 1] = ll.cvt_rn_satfinite_e4m3_f32(hi)


@lk.ll_kernel(backend="cuda", is_entry=True)
def hadd2_kernel(a: ll.Tensor[ll.uint32], b: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    out[t] = ll.add_bf16x2(a[t], b[t])


@requires_cuda
def test_v2_stores_and_fp8_cvt_and_hadd2():
    torch.manual_seed(5)
    # v2 store round trips (64 threads x 2 words = 128 uint32)
    vals = torch.randint(-2**31, 2**31 - 1, (128, ), dtype=torch.int32, device="cuda")
    dst = torch.zeros_like(vals)
    dst_cs = torch.zeros_like(vals)
    kern = v2_store_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(64, 1, 1))
    kern(vals, dst, dst_cs)
    kern.synchronize()
    assert torch.equal(dst, vals), "v2 default store mismatch"
    assert torch.equal(dst_cs, vals), "v2 .cs store mismatch"

    # e4m3 saturating RNE conversions: in-range sweep bitwise vs torch,
    # plus explicit satfinite/NaN edge semantics
    edges = torch.tensor([
        0.0, -0.0, 448.0, -448.0, 447.9, 1.0e-8, -1.0e-8, 2.0**-9, -(2.0**-9), 100.0, -100.0, 1.5, -1.5, 0.02197265625,
        float("nan"),
        float("inf"), -float("inf"), 500.0, -500.0, 448.0001, -448.0001
    ])
    src = torch.cat([torch.empty(107).uniform_(-448.0, 448.0), edges]).cuda()
    assert src.numel() == 128
    packed = torch.zeros(64, dtype=torch.int32, device="cuda")
    byt = torch.zeros(128, dtype=torch.uint8, device="cuda")
    kern2 = quant_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(64, 1, 1))
    kern2(src, packed, byt)
    kern2.synchronize()

    ref_bits = src.cpu().to(torch.float8_e4m3fn).view(torch.uint8)
    in_range = torch.isfinite(src.cpu()) & (src.cpu().abs() <= 448.0)
    assert torch.equal(byt.cpu()[in_range], ref_bits[in_range]), ("in-range e4m3 scalar cvt != torch float8_e4m3fn")
    # NaN: PTX satfinite gives canonical 0x7f; torch gives 0x7f as well
    nan_mask = src.cpu().isnan()
    assert torch.equal(byt.cpu()[nan_mask], torch.full_like(ref_bits[nan_mask], 0x7F)), "NaN cvt"
    # satfinite: |x| > 448 (and +-inf) clamp to the max finite e4m3 magnitude
    pos_sat = (src.cpu() > 448.0) & ~nan_mask
    neg_sat = (src.cpu() < -448.0) & ~nan_mask
    assert torch.equal(byt.cpu()[pos_sat], torch.full_like(byt.cpu()[pos_sat], 0x7E)), \
        (src.cpu()[pos_sat], byt.cpu()[pos_sat])
    assert torch.equal(byt.cpu()[neg_sat], torch.full_like(byt.cpu()[neg_sat], 0xFE)), \
        (src.cpu()[neg_sat], byt.cpu()[neg_sat])
    # packed form: low byte = lo cvt, high byte = hi cvt, upper 16 bits zero
    lo_b, hi_b = byt.cpu()[0::2], byt.cpu()[1::2]
    expect = lo_b.to(torch.int32) | (hi_b.to(torch.int32) << 8)
    assert torch.equal(packed.cpu(), expect), "e4m3x2 packing order mismatch"

    # hadd2: bf16-representable pairs, bitwise vs widen-f32-add-round-once
    a_bf = torch.empty(64, 2).uniform_(-8.0, 8.0).to(torch.bfloat16)
    b_bf = torch.empty(64, 2).uniform_(-8.0, 8.0).to(torch.bfloat16)
    bits = lambda t: (t.view(torch.uint16).to(torch.int32)[:, 0] | (t.view(torch.uint16).to(torch.int32)[:, 1] << 16))
    a_u = bits(a_bf).cuda()
    b_u = bits(b_bf).cuda()
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    kern3 = hadd2_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(64, 1, 1))
    kern3(a_u, b_u, out)
    kern3.synchronize()
    ref = bits((a_bf.float() + b_bf.float()).to(torch.bfloat16))
    assert torch.equal(out.cpu(), ref), "hadd2 != widen-f32-add-RNE reference"


@requires_cuda
def test_atomic_exch_and_segmented_shuffle():
    flags = torch.zeros(8, dtype=torch.int32, device="cuda")
    first_old = torch.zeros(8, dtype=torch.int32, device="cuda")
    kern = exch_kernel.build(PASSES["cuda"], codegen_cuda, grid=(2, 1, 1), block=(256, 1, 1))
    kern(flags, first_old)
    kern.synchronize()
    # every flag ends holding 7; exactly one exch per flag observed the 0 init
    assert torch.equal(flags, torch.full_like(flags, 7)), flags
    assert torch.equal(first_old, torch.ones_like(first_old)), first_old

    out = torch.zeros(32, dtype=torch.int32, device="cuda")
    kern2 = seg_shfl_kernel.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(32, 1, 1))
    kern2(out)
    kern2.synchronize()
    lanes = torch.arange(32, dtype=torch.int32)
    ref = torch.where(lanes % 16 != 15, lanes + 1, lanes)
    assert torch.equal(out.cpu(), ref), (out.cpu(), ref)


if __name__ == "__main__":
    test_vector_ldst_policies()
    print("PASS test_vector_ldst_policies")
    test_v2_stores_and_fp8_cvt_and_hadd2()
    print("PASS test_v2_stores_and_fp8_cvt_and_hadd2")
    test_element_atomics()
    print("PASS test_element_atomics")
    test_scalar_cs_bf16_store_and_prefetch()
    print("PASS test_scalar_cs_bf16_store_and_prefetch")
    test_atomic_exch_and_segmented_shuffle()
    print("PASS test_atomic_exch_and_segmented_shuffle")
