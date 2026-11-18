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
"""Exercise host TMA descriptor APIs against a real Hopper-or-newer driver.

These tests need a CUDA device but do not require Torch or launch GPU kernels.
They check Driver acceptance, address rebinding, and descriptor ownership.
"""

import ctypes

import pytest

import little_kernel.backends.nv.common.runtime.tma_descriptor as tma


@pytest.fixture
def device_allocation():
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip("CUDA Driver not available")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
        "cuCtxPushCurrent_v2": [ctypes.c_void_p],
        "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
        "cuMemAlloc_v2": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "cuMemFree_v2": [ctypes.c_void_p],
    }
    for name, args in signatures.items():
        function = getattr(driver, name)
        function.argtypes = args
        function.restype = ctypes.c_int
    if driver.cuInit(0) != 0:
        pytest.skip("CUDA Driver initialization unavailable")
    count = ctypes.c_int()
    assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0
    if not count.value:
        pytest.skip("No CUDA devices")
    major = ctypes.c_int()
    assert driver.cuDeviceGetAttribute(ctypes.byref(major), 75, 0) == 0
    if major.value < 9:
        pytest.skip("TMA requires compute capability 9.0 or newer")
    context = ctypes.c_void_p()
    assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), 0) == 0
    try:
        assert driver.cuCtxPushCurrent_v2(context) == 0
        try:
            address = ctypes.c_void_p()
            assert driver.cuMemAlloc_v2(ctypes.byref(address), 131072) == 0
            try:
                yield address.value
            finally:
                assert driver.cuMemFree_v2(address) == 0
        finally:
            popped = ctypes.c_void_p()
            assert driver.cuCtxPopCurrent_v2(ctypes.byref(popped)) == 0
    finally:
        assert driver.cuDevicePrimaryCtxRelease_v2(0) == 0


@pytest.mark.parametrize("inner,outer,box_inner", [
    (256, 64, 16),
    (256, 64, 32),
    (256, 64, 64),
    # A boundary box may exceed either global dimension.
    (32, 16, 64),
])
def test_tma_plan_encodes_and_rebases_with_real_driver(device_allocation, inner, outer, box_inner):
    plan = tma.TiledTensorMap2DPlan(
        global_address=device_allocation,
        global_inner_dim=inner,
        global_outer_dim=outer,
        box_inner_dim=box_inner,
        box_outer_dim=64,
        outer_stride_bytes=inner * 2,
        data_type=tma.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        element_size_bytes=2,
    )
    descriptor = tma.get_tma_helper().encode_tiled_2d_plan(plan)
    before = bytes(descriptor.data)
    rebound = tma.rebased_cu_tensor_map(descriptor, device_allocation + 65536)
    # Compare opaque values without interpreting their implementation layout.
    assert bytes(descriptor.data) == before
    assert bytes(rebound.data) != before
    assert ctypes.addressof(descriptor) % 64 == 0
    assert ctypes.addressof(rebound) % 64 == 0
