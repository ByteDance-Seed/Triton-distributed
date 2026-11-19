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
"""Review regressions for pointer narrowing and extended TMA swizzle spans."""

import ctypes
import gc
from types import SimpleNamespace

import pytest

import little_kernel.backends.nv.common.runtime.tma_descriptor as tma


def _plan(**updates):
    fields = dict(
        global_address=0x10000,
        global_inner_dim=256,
        global_outer_dim=64,
        box_inner_dim=64,
        box_outer_dim=64,
        outer_stride_bytes=512,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )
    fields.update(updates)
    return tma.TiledTensorMap2DPlan(**fields)


@pytest.mark.parametrize("extra", [0, 16])
def test_plan_rejects_address_that_would_wrap_to_another_device_pointer(extra):
    address = (1 << (8 * ctypes.sizeof(tma.CUdeviceptr))) + extra
    with pytest.raises(ValueError, match="global_address"):
        _plan(global_address=address)


@pytest.mark.parametrize("swizzle", [
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
])
def test_extended_128byte_swizzle_rejects_a_256byte_inner_box(swizzle):
    # The plan validates byte spans independently of architecture support.
    # The Driver remains responsible for device-specific swizzle support.
    with pytest.raises(ValueError, match="swizzle"):
        _plan(swizzle=swizzle, box_inner_dim=128)


@pytest.mark.parametrize("swizzle", [
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
    tma.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
])
def test_extended_128byte_swizzle_allows_a_smaller_inner_box(swizzle):
    assert _plan(swizzle=swizzle, box_inner_dim=32).box_inner_dim == 32


def test_multiple_encoded_descriptors_retain_distinct_aligned_storage():
    helper = tma.TMADescriptorHelper.__new__(tma.TMADescriptorHelper)
    helper._libcuda = SimpleNamespace(cuTensorMapEncodeTiled=lambda *args: 0)
    descriptors = [helper.encode_tiled_2d_plan(_plan()) for _ in range(32)]
    gc.collect()
    addresses = [ctypes.addressof(descriptor) for descriptor in descriptors]
    assert len(set(addresses)) == len(descriptors)
    assert all(address % 64 == 0 for address in addresses)
