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
"""RF12 canonical semantic fixtures for the 16-byte global vector families.

Three independently runnable stages share the *same executable fixtures* from
:mod:`examples.nv.common.testing.vector_fixtures`; none of them uses benchmark-source
string rewriting as an oracle:

1. **Geometry invariants** (always): ``nvec`` is a vector count, not an
   element count, tails are explicit and bounds are checked.
2. **CPU codegen** (always): generated CUDA carries the real cache-policy
   intrinsics (``__ldcg`` / ``__ldcs`` / ``__ldg`` / ``__stcg`` / ``__stcs``),
   helper definitions, ABI include and ordering contracts.  Literal PTX
   mnemonics are asserted by the NVCC stage below, since nvcc lowers these
   intrinsics to ``ld.global.*`` / ``st.global.*``.
3. **NVCC** (only when ``nvcc`` is on PATH): explicit ``sm_90a`` PTX
   cross-compilation; the emitted PTX mnemonics are inspected but nothing
   is launched.
4. **GPU semantics** (only on a real CUDA GPU, capability >= 9.0): bit-exact
   round trips, masked-tail copies and packed-lane order canaries.  These
   tests skip with a stated capability requirement on CPU workers; a skip
   is never counted as a GPU pass.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

# Make the runnable example fixtures importable when pytest is invoked from
# the repository root (the examples tree is deliberately outside the package).
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.nv.common.testing import vector_fixtures as vf
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES

THREADS = vf.THREADS_PER_VECTOR_CTA
NVCC_ARCH = "sm_90a"

has_nvcc = shutil.which("nvcc") is not None
requires_nvcc = pytest.mark.skipif(not has_nvcc, reason="nvcc not on PATH")

try:  # GPU stage is optional and explicitly gated; CPU stages never import it.
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None

_gpu_capability = None
if torch is not None:
    try:
        if torch.cuda.is_available():
            _gpu_capability = torch.cuda.get_device_capability(0)
    except Exception:  # pragma: no cover - environment dependent
        _gpu_capability = None

requires_sm90_gpu = pytest.mark.skipif(
    _gpu_capability is None or _gpu_capability < (9, 0),
    reason=("RF12 GPU semantic stage requires a CUDA GPU with compute capability "
            ">= 9.0 (sm_90 class); unavailable here so the round trip is not "
            "counted as passed"),
)


def _cuda(kernel) -> str:
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=THREADS)


# ---------------------------------------------------------------------------
# 1. Geometry: vector count / tail policy (no GPU, no compiler)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", vf.U8X16_CASES + vf.BF16X8_CASES)
def test_geometry_uses_vector_count_and_is_in_bounds(geometry):
    assert geometry.nvec == vf.ceil_div(geometry.nelem, geometry.lanes)
    # The reviewed defect: element count used where a vector count is
    # required.  A correct vector loop never reads/writes past allocation.
    assert geometry.vector_bounds_are_safe()
    assert geometry.nvec <= THREADS, "fixture block must cover every vector"
    if geometry.tail_policy == "masked":
        assert geometry.nelem % geometry.lanes != 0
        # The scalar tail owns exactly the ragged lanes.
        assert geometry.allocated_elems == geometry.nelem
    else:
        assert geometry.allocated_elems == geometry.nvec * geometry.lanes


def test_the_reviewed_element_count_fixture_would_be_out_of_bounds():
    """Document the original defect as a negative invariant.

    With ``nvec = nelem`` (element count) and ``off = tid * lanes`` the last
    thread reaches byte ``nelem * lanes``; for u8x16 that is 16x past the end.
    The shared geometry helper must never describe such a launch.
    """
    nelem = 32
    lanes = vf.U8X16_LANES
    bogus_last_offset_end = nelem * lanes  # what the broken fixture computed
    allocation = nelem  # it only allocated nelem elements
    assert bogus_last_offset_end > allocation
    with pytest.raises(ValueError, match="masked tail policy requires"):
        # A non-ragged nelem cannot pretend to be a masked tail case.
        vf.VectorGeometry(lanes, lanes, "masked")
    with pytest.raises(ValueError, match="padded tail policy requires"):
        vf.VectorGeometry(lanes + 1, lanes, "padded")
    with pytest.raises(ValueError, match="unsupported tail policy"):
        vf.VectorGeometry(lanes, lanes, "guess")


# ---------------------------------------------------------------------------
# 2. CPU codegen: real PTX opcodes, helpers, ABI include, ordering
# ---------------------------------------------------------------------------


def test_u8x16_one_and_several_vector_kernel_emits_all_policy_intrinsics():
    cuda = _cuda(vf.copy_u8x16_policies)
    # CUDA cache-policy intrinsics the opaque-byte family documents; nvcc
    # lowers these to ld/st.global.{default,cg,cs}.v4.u32 (checked below).
    assert "*reinterpret_cast<const uint4*>(ptr)" in cuda
    assert "__ldcg(reinterpret_cast<const uint4*>(ptr))" in cuda
    assert "__ldcs(reinterpret_cast<const uint4*>(ptr))" in cuda
    assert "__stcg(reinterpret_cast<uint4*>(ptr), value)" in cuda
    assert "__stcs(reinterpret_cast<uint4*>(ptr), value)" in cuda
    # Helpers are defined once even though the kernel uses each policy once.
    for helper in (
            "void gmem_ld_u8x16_fn(",
            "void gmem_ld_u8x16_cg_fn(",
            "void gmem_ld_u8x16_streaming_fn(",
            "void gmem_st_u8x16_fn(",
            "void gmem_st_u8x16_cg_fn(",
            "void gmem_st_u8x16_streaming_fn(",
    ):
        assert cuda.count(helper) == 1, helper
    # r0 carries bytes 0-7 and r1 carries bytes 8-15 (packed-lane order).
    assert "(uint64_t)value.x | ((uint64_t)value.y << 32)" in cuda
    # Opaque byte move: never a numeric conversion or fp8/bf16 include.
    assert "cvt.rn.satfinite" not in cuda
    assert "<cuda_bf16.h>" not in cuda


def test_u8x16_nc_path_uses_noncoherent_intrinsic():
    cuda = _cuda(vf.copy_u8x16_nc)
    assert "__ldg(reinterpret_cast<const uint4*>(ptr))" in cuda
    assert "__ldcg(" not in cuda
    assert "__ldcs(" not in cuda


def test_u8x16_lane_canary_writes_every_lane_to_its_own_slot_in_memory_order():
    cuda = _cuda(vf.u8x16_lane_canary)
    assert "uint32_t vec_u8x16_lane_fn(" in cuda
    # Each lane extractor call carries its compile-time literal lane 0..15;
    # a dynamic lane can never reach this family (register pair is not RAM).
    calls = re.findall(r"vec_u8x16_lane_fn\(lo, hi, (\d+)\)", cuda)
    assert sorted(int(n) for n in calls) == list(range(16))
    # Exactly one opaque 16-byte fetch feeds all sixteen lane extracts.
    assert "gmem_ld_u8x16_fn(x + off" in cuda
    # Order-sensitive oracle: each extracted byte k is written to its OWN
    # slot lane_order[off + k] (never folded into a commutative sum).
    for lane in range(16):
        assert f"lane_order[off + {lane}]" in cuda
    assert "% 256" not in cuda


def test_u8x16_masked_tail_never_issues_a_partial_vector():
    cuda = _cuda(vf.copy_u8x16_masked_tail)
    vector_store_pos = cuda.index("gmem_st_u8x16_fn(dst + off")
    full_branch = cuda[cuda.rindex("if", 0, vector_store_pos):vector_store_pos]
    # The full-vector branch is guarded by (tid + 1) < nvec, so the last
    # partial vector can never reach a 16-byte transaction.
    assert "tid + 1 < nvec" in full_branch
    tail_guard_pos = cuda.index("if (idx < nelem)")
    # The scalar tail comes after the vector branch and uses scalar moves.
    assert tail_guard_pos > vector_store_pos
    assert "x[idx]" in cuda and "dst[idx]" in cuda
    # Only one vector helper call site each for load and store.
    assert cuda.count("gmem_ld_u8x16_fn(x + off") == 1
    assert cuda.count("gmem_st_u8x16_fn(dst + off") == 1


def test_bf16x8_roundtrip_emits_rne_pack_and_policy_stores():
    cuda = _cuda(vf.copy_bf16x8_roundtrip)
    assert "__stcg(reinterpret_cast<uint4*>(ptr), value)" in cuda
    assert "__stcs(reinterpret_cast<uint4*>(ptr), value)" in cuda
    # Eight lanes widened, then RNE-packed; the bf16 ABI include is emitted.
    assert "__floats2bfloat162_rn(f0, f1)" in cuda
    assert "<cuda_bf16.h>" in cuda
    # Every literal lane 0..7 is extracted in order.
    calls = re.findall(r"vec_bf16x8_lane_fn\(lo, hi, (\d+)\)", cuda)
    assert sorted(int(n) for n in calls) == list(range(8))
    # Default/.cg/.cs packed stores are each called exactly once.
    assert cuda.count("gmem_st_bf16x8_fn(dst + off") == 1
    assert cuda.count("gmem_st_bf16x8_cg_fn(dst_cg + off") == 1
    assert cuda.count("gmem_st_bf16x8_streaming_fn(dst_cs + off") == 1


def test_bf16x8_masked_tail_uses_scalar_rne_store_not_a_vector():
    cuda = _cuda(vf.copy_bf16x8_masked_tail)
    assert "st_global_cs_bf16_f32_rn_fn(" in cuda
    assert "if (tid + 1 < nvec)" in cuda
    assert "if (idx < nelem)" in cuda
    # The vector pack exists only in the full-vector branch (one call site).
    assert cuda.count("gmem_st_bf16x8_fn(dst + off") == 1


def test_bf16x8_widening_policy_paths_use_ldg_ldcs():
    cuda = _cuda(vf.copy_bf16x8_f32x8_policy)
    assert "__ldg(reinterpret_cast<const uint4*>(ptr))" in cuda
    assert "__ldcs(reinterpret_cast<const uint4*>(ptr))" in cuda
    assert "__bfloat162float(p[0].x)" in cuda
    assert "void gmem_ld_bf16x8_f32x8_nc_fn(" in cuda
    assert "void gmem_ld_bf16x8_f32x8_cs_fn(" in cuda
    # Both widened paths are committed through streaming stores.
    assert cuda.count("gmem_st_bf16x8_streaming_fn(") >= 2


# ---------------------------------------------------------------------------
# 3. NVCC stage: explicit sm_90a PTX cross-compilation (never launches)
# ---------------------------------------------------------------------------


def _ptx(kernel, name: str) -> str:
    from little_kernel.backends.nv.common.runtime.compiler import compile_cuda

    return compile_cuda(_cuda(kernel), name, arch=NVCC_ARCH, target_format="ptx").decode()


@requires_nvcc
@pytest.mark.parametrize(
    "kernel,name",
    [
        (vf.copy_u8x16_policies, "u8x16_policies"),
        (vf.copy_u8x16_nc, "u8x16_nc"),
        (vf.u8x16_lane_canary, "u8x16_canary"),
        (vf.copy_u8x16_masked_tail, "u8x16_tail"),
        (vf.copy_bf16x8_roundtrip, "bf16x8_roundtrip"),
        (vf.copy_bf16x8_masked_tail, "bf16x8_tail"),
        (vf.copy_bf16x8_f32x8_policy, "bf16x8_policy"),
    ],
)
def test_fixtures_cross_compile_to_sm90a_ptx(kernel, name):
    ptx = _ptx(kernel, name)
    assert ".target sm_90a" in ptx
    assert ptx.strip()


@requires_nvcc
def test_nvcc_ptx_carries_exact_vector_opcodes_and_scalar_tail():
    ptx = _ptx(vf.copy_u8x16_policies, "u8x16_policies")
    for opcode in (
            "ld.global.v4.u32",
            "ld.global.cg.v4.u32",
            "ld.global.cs.v4.u32",
            "st.global.v4.u32",
            "st.global.cg.v4.u32",
            "st.global.cs.v4.u32",
    ):
        assert opcode in ptx, opcode

    tail = _ptx(vf.copy_u8x16_masked_tail, "u8x16_tail")
    # Exactly one full vector transaction; the ragged lanes are scalar.
    assert tail.count("ld.global.v4.u32") == 1
    assert "ld.global.u8" in tail and "st.global.u8" in tail
    # No second vector transaction may cover the partial tail.
    assert tail.count("st.global.v4.u32") == 1


@requires_nvcc
def test_nvcc_bf16_tail_uses_scalar_u16_rne_store():
    ptx = _ptx(vf.copy_bf16x8_masked_tail, "bf16x8_tail")
    assert ptx.count("ld.global.v4.u32") == 1
    assert "ld.global.u16" in ptx
    assert "st.global.cs.u16" in ptx


# ---------------------------------------------------------------------------
# 4. GPU semantic stage: explicit capability gate (not run on CPU workers)
# ---------------------------------------------------------------------------


def _bf16_round_trip(values):
    """NumPy BF16 RNE reference, independent of the kernel implementation."""
    import numpy as np

    bits = values.astype(np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (rounded.astype(np.uint32) << 16).view(np.float32), rounded


@requires_sm90_gpu
def test_gpu_u8x16_roundtrip_and_lane_canary():
    import numpy as np

    for geometry in vf.U8X16_CASES:
        nvec = geometry.nvec
        allocated = geometry.allocated_elems
        host = (np.arange(allocated, dtype=np.int64) * 7 + 13) % 256
        host = host.astype(np.uint8)
        src = torch.from_numpy(host).cuda()
        outs = {p: torch.zeros_like(src) for p in ("d", "cg", "cs", "nc")}

        if geometry.tail_policy == "padded":
            kern = vf.copy_u8x16_policies.build(
                PASSES["cuda"],
                codegen_cuda,
                grid=(1, 1, 1),
                block=(THREADS, 1, 1),
            )
            kern(src, outs["d"], outs["cg"], outs["cs"], nvec)
            kern.synchronize()
            nc = vf.copy_u8x16_nc.build(
                PASSES["cuda"],
                codegen_cuda,
                grid=(1, 1, 1),
                block=(THREADS, 1, 1),
            )
            nc(src, outs["nc"], nvec)
            nc.synchronize()
            for out in outs.values():
                assert torch.equal(out.cpu(), torch.from_numpy(host)), geometry
        else:
            kern = vf.copy_u8x16_masked_tail.build(
                PASSES["cuda"],
                codegen_cuda,
                grid=(1, 1, 1),
                block=(THREADS, 1, 1),
            )
            kern(src, outs["d"], nvec, geometry.nelem)
            kern.synchronize()
            assert torch.equal(outs["d"].cpu(), torch.from_numpy(host)), geometry

    # Lane-order canary on the one-vector case: distinct literal bytes
    # 0..15 written one-per-slot, compared as a full ordered sequence.
    host = np.arange(16, dtype=np.uint8)
    src = torch.from_numpy(host).cuda()
    lane_order = torch.zeros(16, dtype=torch.uint8, device="cuda")
    kern = vf.u8x16_lane_canary.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(16, 1, 1))
    kern(src, lane_order, 1)
    kern.synchronize()
    assert torch.equal(lane_order.cpu(), torch.from_numpy(host))

    # Negative permutation check: a reversed (half-swapped) input must come
    # out reversed and therefore DIFFER from in-order expected.  A sum-based
    # oracle could not distinguish this; the per-slot ordering oracle does.
    reversed_host = host[::-1].copy()
    rev_src = torch.from_numpy(reversed_host).cuda()
    rev_out = torch.zeros(16, dtype=torch.uint8, device="cuda")
    kern(rev_src, rev_out, 1)
    kern.synchronize()
    assert torch.equal(rev_out.cpu(), torch.from_numpy(reversed_host))
    assert not torch.equal(rev_out.cpu(), torch.from_numpy(host))


@requires_sm90_gpu
def test_gpu_bf16x8_roundtrip_is_bitexact_under_every_policy():
    import numpy as np

    geometry = vf.BF16X8_CASES[1]  # several full vectors
    nvec = geometry.nvec
    rng = np.random.default_rng(0)
    f32 = rng.uniform(-100, 100, size=geometry.allocated_elems).astype(np.float32)
    ref_f32, ref_u16 = _bf16_round_trip(f32)
    src = torch.from_numpy(ref_f32).to(torch.bfloat16).cuda()
    dst = torch.zeros_like(src)
    dst_cg = torch.zeros_like(src)
    dst_cs = torch.zeros_like(src)
    kern = vf.copy_bf16x8_roundtrip.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(THREADS, 1, 1))
    kern(src, dst, dst_cg, dst_cs, nvec)
    kern.synchronize()
    for out in (dst, dst_cg, dst_cs):
        got = out.cpu().view(torch.uint16).numpy()
        assert np.array_equal(got, ref_u16), "bf16x8 RNE round trip mismatch"


@requires_sm90_gpu
def test_gpu_bf16x8_masked_tail_and_widening_policies():
    import numpy as np

    geometry = vf.BF16X8_CASES[2]  # ragged tail
    nvec, nelem = geometry.nvec, geometry.nelem
    rng = np.random.default_rng(1)
    f32 = rng.uniform(-50, 50, size=nelem).astype(np.float32)
    _, ref_u16 = _bf16_round_trip(f32)
    src = torch.from_numpy((ref_u16.astype(np.uint32) << 16).view(np.float32)).to(torch.bfloat16).cuda()
    dst = torch.zeros(nelem, dtype=torch.bfloat16, device="cuda")
    kern = vf.copy_bf16x8_masked_tail.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(THREADS, 1, 1))
    kern(src, dst, nvec, nelem)
    kern.synchronize()
    assert np.array_equal(dst.cpu().view(torch.uint16).numpy(), ref_u16), "bf16x8 masked-tail mismatch"

    # Policy widening round trip on full vectors.
    full = vf.BF16X8_CASES[1]
    rng2 = np.random.default_rng(2)
    wide_f32 = rng2.uniform(-50, 50, size=full.allocated_elems).astype(np.float32)
    _, wide_u16 = _bf16_round_trip(wide_f32)
    wide = torch.from_numpy((wide_u16.astype(np.uint32) << 16).view(np.float32)).to(torch.bfloat16).cuda()
    out_nc = torch.zeros_like(wide)
    out_cs = torch.zeros_like(wide)
    kern = vf.copy_bf16x8_f32x8_policy.build(PASSES["cuda"], codegen_cuda, grid=(1, 1, 1), block=(THREADS, 1, 1))
    kern(wide, out_nc, out_cs, full.nvec)
    kern.synchronize()
    assert torch.equal(out_nc.cpu(), wide.cpu())
    assert torch.equal(out_cs.cpu(), wide.cpu())
