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
"""CPU regression coverage for reusable Hopper grouped-GEMM building blocks.

These tests intentionally do not initialize CUDA or load libcuda.  They cover
host TMA plan/address transforms, full/empty mbarrier ring schedule
arithmetic, the m64n256k16 FP32 accumulator fragment map, and codegen of the
FP16 SS WGMMA with MN-major B (trans_b=1).
"""

import ctypes
from types import SimpleNamespace

import pytest

import little_kernel.language as ll
from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma_intrin
from little_kernel.backends.nv.common.language.intrin.barrier import MBarrierFullEmptyRing
import little_kernel.backends.nv.common.runtime.tma_descriptor as tma


def test_grouped_tma_plans_rebase_runtime_addresses():
    template = tma.TiledTensorMap2DPlan(
        global_address=0,
        global_inner_dim=256,
        global_outer_dim=64,
        box_inner_dim=64,
        box_outer_dim=64,
        outer_stride_bytes=512,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )

    plans = tma.build_grouped_tiled_2d_plans(
        base_address=0x10000,
        group_byte_offsets=(0, 32768, 65536),
        template=template,
    )

    assert [plan.global_address for plan in plans] == [0x10000, 0x10000 + 32768, 0x10000 + 65536]
    assert all(plan.box_inner_dim == 64 for plan in plans)
    assert all(plan.swizzle == tma.CU_TENSOR_MAP_SWIZZLE_128B for plan in plans)


@pytest.mark.parametrize("box_inner", [16, 32, 64])
def test_swizzle128_plan_accepts_boxes_up_to_128_bytes(box_inner):
    plan = tma.TiledTensorMap2DPlan(
        global_address=0x10000,
        global_inner_dim=256,
        global_outer_dim=64,
        box_inner_dim=box_inner,
        box_outer_dim=64,
        outer_stride_bytes=512,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )
    assert plan.box_inner_dim == box_inner


def test_swizzle128_plan_rejects_box_larger_than_swizzle_span():
    with pytest.raises(ValueError, match="swizzle"):
        tma.TiledTensorMap2DPlan(
            global_address=0x10000,
            global_inner_dim=256,
            global_outer_dim=64,
            box_inner_dim=128,
            box_outer_dim=64,
            outer_stride_bytes=512,
            data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            element_size_bytes=2,
        )


def test_tma_plan_supports_boundary_box_larger_than_global_dimensions():
    plan = tma.TiledTensorMap2DPlan(
        global_address=0x10000,
        global_inner_dim=32,
        global_outer_dim=16,
        box_inner_dim=64,
        box_outer_dim=64,
        outer_stride_bytes=64,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )
    assert plan.box_inner_dim > plan.global_inner_dim
    assert plan.box_outer_dim > plan.global_outer_dim


def test_rebased_cu_tensor_map_uses_driver_and_preserves_template(monkeypatch):
    template = tma.CUtensorMap()
    template.data[:] = bytes(range(128))
    calls = []

    def fake_replace(tensor_map, address):
        result = ctypes.cast(tensor_map, ctypes.POINTER(tma.CUtensorMap)).contents
        assert ctypes.addressof(result) % 64 == 0
        assert bytes(result.data) == bytes(template.data)
        calls.append(address.value)
        # Deliberately use a different offset from the rejected implementation.
        # Only the driver knows where/how an address is encoded in the payload.
        result.data[80:88] = address.value.to_bytes(8, "little")
        return 0

    helper = tma.TMADescriptorHelper.__new__(tma.TMADescriptorHelper)
    helper._libcuda = SimpleNamespace(cuTensorMapReplaceAddress=fake_replace)
    monkeypatch.setattr(tma, "get_tma_helper", lambda: helper)

    rebased = tma.rebased_cu_tensor_map(template, 0x20000)

    assert calls == [0x20000]
    assert bytes(template.data) == bytes(range(128))
    expected = bytearray(range(128))
    expected[80:88] = (0x20000).to_bytes(8, "little")
    assert bytes(rebased.data) == expected


def test_rebased_cu_tensor_map_propagates_driver_error(monkeypatch):
    template = tma.CUtensorMap()
    template.data[:] = bytes(range(128))

    def fake_replace(tensor_map, address):
        ctypes.cast(tensor_map, ctypes.POINTER(tma.CUtensorMap)).contents.data[0] = 0
        return 1

    def fake_error_string(error, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_char_p))[0] = b"invalid address"
        return 0

    helper = tma.TMADescriptorHelper.__new__(tma.TMADescriptorHelper)
    helper._libcuda = SimpleNamespace(cuTensorMapReplaceAddress=fake_replace, cuGetErrorString=fake_error_string)
    monkeypatch.setattr(tma, "get_tma_helper", lambda: helper)

    with pytest.raises(RuntimeError, match="cuTensorMapReplaceAddress: invalid address"):
        tma.rebased_cu_tensor_map(template, 0x20000)
    assert bytes(template.data) == bytes(range(128))


@pytest.mark.parametrize("address,error", [
    (-16, ValueError),
    (17, ValueError),
    (1 << (8 * ctypes.sizeof(tma.CUdeviceptr)), ValueError),
    (True, TypeError),
    (16.0, TypeError),
])
def test_rebased_cu_tensor_map_validates_before_loading_driver(monkeypatch, address, error):

    def unexpected_driver_load():
        pytest.fail("invalid device addresses must be rejected before loading CUDA")

    monkeypatch.setattr(tma, "get_tma_helper", unexpected_driver_load)
    with pytest.raises(error, match="global_address"):
        tma.rebased_cu_tensor_map(tma.CUtensorMap(), address)


def test_encode_grouped_plans_call_driver_without_gpu():
    template = tma.TiledTensorMap2DPlan(
        global_address=0,
        global_inner_dim=256,
        global_outer_dim=64,
        box_inner_dim=64,
        box_outer_dim=64,
        outer_stride_bytes=512,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )
    calls = []

    def fake_encode(tensor_map, dtype, rank, address, dims, strides, box, element_strides, interleave, swizzle, l2,
                    oob):
        assert ctypes.addressof(ctypes.cast(tensor_map, ctypes.POINTER(tma.CUtensorMap)).contents) % 64 == 0
        calls.append({
            "dtype": dtype,
            "rank": rank,
            "address": address.value,
            "inner": dims[0],
            "outer": dims[1],
            "stride": strides[0],
            "box_inner": box[0],
            "box_outer": box[1],
            "swizzle": swizzle,
        })
        return 0

    helper = tma.TMADescriptorHelper.__new__(tma.TMADescriptorHelper)
    helper._libcuda = SimpleNamespace(cuTensorMapEncodeTiled=fake_encode)

    descriptors = helper.encode_grouped_tiled_2d(0x10000, (0, 32768), template)

    assert len(descriptors) == 2
    assert [call["address"] for call in calls] == [0x10000, 0x18000]
    assert calls[0] == {
        "dtype": tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        "rank": 2,
        "address": 0x10000,
        "inner": 256,
        "outer": 64,
        "stride": 512,
        "box_inner": 64,
        "box_outer": 64,
        "swizzle": tma.CU_TENSOR_MAP_SWIZZLE_128B,
    }


def test_four_stage_mbarrier_ring_parities_and_slots():
    ring = MBarrierFullEmptyRing(stages=4, transaction_bytes=40960, consumer_arrivals=128)

    assert ring.consumer_full_wait(0).slot == 0
    assert ring.consumer_full_wait(3).phase == 0
    assert ring.consumer_full_wait(4) == ring.consumer_full_wait(4)
    assert ring.consumer_full_wait(7) == ll.MBarrierRingSlot(3, 1)

    assert ring.producer_empty_wait(3) is None
    assert ring.producer_empty_wait(4) == ll.MBarrierRingSlot(0, 0)
    assert ring.producer_empty_wait(8) == ll.MBarrierRingSlot(0, 1)

    assert ring.consumer_release_previous(0) is None
    assert ring.consumer_release_previous(1) == 0
    assert ring.consumer_release_previous(4) == 3
    assert ring.consumer_release_final(7) == 3


def test_mbarrier_ring_rejects_illegal_configuration():
    with pytest.raises(ValueError, match="stages"):
        MBarrierFullEmptyRing(stages=9, transaction_bytes=4096)
    with pytest.raises(ValueError, match="16-byte"):
        MBarrierFullEmptyRing(stages=4, transaction_bytes=4095)


def test_m64n256_fragment_coordinate_known_points():
    coord = ll.wgmma_m64n256k16_accumulator_coordinate
    assert coord(0, 0) == (0, 0)
    assert coord(0, 1) == (0, 1)
    assert coord(0, 2) == (8, 0)
    assert coord(0, 4) == (0, 8)
    assert coord(1, 0) == (0, 2)
    assert coord(4, 0) == (1, 0)
    assert coord(32, 0) == (16, 0)


def test_m64n256_fragment_map_is_a_bijection():
    coord = ll.wgmma_m64n256k16_accumulator_coordinate
    covered = set()
    for tid in range(128):
        for reg in range(128):
            row, col = coord(tid, reg)
            assert 0 <= row < 64
            assert 0 <= col < 256
            covered.add((row, col))
            if reg % 2 == 0:
                assert coord(tid, reg + 1) == (row, col + 1)
    assert len(covered) == 64 * 256


def test_m64n256_fragment_rejects_out_of_range_inputs():
    coord = ll.wgmma_m64n256k16_accumulator_coordinate
    with pytest.raises(ValueError):
        coord(128, 0)
    with pytest.raises(ValueError):
        coord(0, 128)


def test_m64n256k16_f16_ss_codegen_uses_trans_b():
    codegen_func = getattr(ll.wgmma_m64n256k16_f16, CODEGEN_FUNC_ATTR)
    builtin = codegen_func("acc", "desc_a", "desc_b", 0, 1, 1)

    assert "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16" in builtin.body
    assert "p,1,1,0,1;" in builtin.body
    assert builtin.body.count('"+f"(c[') == 128
    assert "wgmma_m64n256k16_f16_a0b1d1_fn(acc, desc_a, desc_b)" in builtin.return_val
