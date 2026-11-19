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
"""Torch-free compiler -> CUDA -> nvcc -> real SM90 numerical smoke test."""
import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES

pytestmark = [pytest.mark.gpu, pytest.mark.sm90]


@lk.ll_kernel(backend='cuda', is_entry=True)
def ci_vector_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
                  n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = a[i] + b[i]


@pytest.fixture(scope='module', params=range(int(os.environ.get('LK_CI_DEVICE_COUNT', '1'))))
def cuda(request):
    try:
        driver = ctypes.CDLL('libcuda.so.1')
    except OSError:
        pytest.skip('CUDA Driver not available; GPU correctness is unverified')
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    uint = ctypes.c_uint
    functions = {
        'cuInit': [uint],
        'cuDeviceGetCount': [ctypes.POINTER(integer)],
        'cuDeviceGetAttribute': [ctypes.POINTER(integer), integer, integer],
        'cuDevicePrimaryCtxRetain': [ctypes.POINTER(pointer), integer],
        'cuDevicePrimaryCtxRelease_v2': [integer],
        'cuCtxPushCurrent_v2': [pointer],
        'cuCtxPopCurrent_v2': [ctypes.POINTER(pointer)],
        'cuCtxSynchronize': [],
        'cuMemAlloc_v2': [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        'cuMemFree_v2': [ctypes.c_uint64],
        'cuMemcpyHtoD_v2': [ctypes.c_uint64, pointer, ctypes.c_size_t],
        'cuMemcpyDtoH_v2': [pointer, ctypes.c_uint64, ctypes.c_size_t],
        'cuModuleLoadData': [ctypes.POINTER(pointer), pointer],
        'cuModuleUnload': [pointer],
        'cuModuleGetFunction': [ctypes.POINTER(pointer), pointer, ctypes.c_char_p],
        'cuLaunchKernel':
        [pointer, uint, uint, uint, uint, uint, uint, uint, pointer,
         ctypes.POINTER(pointer),
         ctypes.POINTER(pointer)],
    }
    for name, args in functions.items():
        function = getattr(driver, name)
        function.argtypes = args
        function.restype = integer
    if driver.cuInit(0) != 0:
        pytest.skip('CUDA initialization unavailable; GPU correctness is unverified')
    count = integer()
    assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0
    if count.value == 0: pytest.skip('No CUDA devices; GPU correctness is unverified')
    expected = int(os.environ.get('LK_CI_DEVICE_COUNT', '1'))
    assert 1 <= expected <= 8 and count.value >= expected
    device = request.param
    major = integer()
    assert driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device) == 0
    if major.value != 9: pytest.skip('The declared H800 lane requires an SM90 device')
    context = pointer()
    assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
    assert driver.cuCtxPushCurrent_v2(context) == 0
    try:
        yield driver
    finally:
        popped = pointer()
        assert driver.cuCtxPopCurrent_v2(ctypes.byref(popped)) == 0
        assert driver.cuDevicePrimaryCtxRelease_v2(device) == 0


@pytest.fixture(scope='module')
def generated_kernel(cuda, tmp_path_factory):
    nvcc = shutil.which('nvcc')
    if not nvcc: pytest.skip('nvcc unavailable; generated CUDA compilation is unverified')
    directory = tmp_path_factory.mktemp('generated-ci-cuda')
    artifact = ci_vector_add.compile_artifact(PASSES['cuda'], codegen_cuda, target='sm_90a')
    source, binary = directory / 'vector_add.cu', directory / 'vector_add.cubin'
    source.write_text(artifact.source)
    subprocess.run([nvcc, '-std=c++17', '-arch=sm_90a', '--cubin',
                    str(source), '-o', str(binary)], check=True, text=True, timeout=120)
    data = ctypes.create_string_buffer(binary.read_bytes())
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    assert cuda.cuModuleLoadData(ctypes.byref(module), data) == 0
    try:
        assert cuda.cuModuleGetFunction(ctypes.byref(function), module, artifact.kernel_name.encode()) == 0
        yield function
    finally:
        assert cuda.cuModuleUnload(module) == 0


@pytest.mark.parametrize('n', [1, 31, 32, 257, 4097])
def test_generated_vector_add_on_device(cuda, generated_kernel, n):
    a = np.arange(n, dtype=np.float32) * np.float32(.25)
    b = np.arange(n, dtype=np.float32) * np.float32(-.125)
    out = np.full(n, -777, dtype=np.float32)
    buffers = []
    try:
        for host in (a, b, out):
            address = ctypes.c_uint64()
            assert cuda.cuMemAlloc_v2(ctypes.byref(address), host.nbytes) == 0
            buffers.append(address)
            assert cuda.cuMemcpyHtoD_v2(address, ctypes.c_void_p(host.ctypes.data), host.nbytes) == 0
        values = [*buffers, ctypes.c_int(n)]
        args = (ctypes.c_void_p * len(values))(*[ctypes.addressof(value) for value in values])
        assert cuda.cuLaunchKernel(generated_kernel, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None) == 0
        assert cuda.cuCtxSynchronize() == 0
        assert cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(out.ctypes.data), buffers[2], out.nbytes) == 0
        np.testing.assert_array_equal(out, a + b)
    finally:
        for address in reversed(buffers):
            assert cuda.cuMemFree_v2(address) == 0
