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
"""Test cudaIpcOpenMemHandle with spawn mode and flag=0."""
import ctypes
import multiprocessing
import time

IPC_H_SIZE = 64


class IpcH(ctypes.Structure):
    _fields_ = [('reserved', ctypes.c_char * IPC_H_SIZE)]


def get_rt():
    rt = ctypes.CDLL('libcudart.so')
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaMalloc.restype = ctypes.c_int
    rt.cudaFree.argtypes = [ctypes.c_void_p]
    rt.cudaFree.restype = ctypes.c_int
    rt.cudaSetDevice.argtypes = [ctypes.c_int]
    rt.cudaSetDevice.restype = ctypes.c_int
    rt.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    rt.cudaMemset.restype = ctypes.c_int
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rt.cudaMemcpy.restype = ctypes.c_int
    rt.cudaDeviceSynchronize.restype = ctypes.c_int
    rt.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    rt.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
    rt.cudaGetLastError.argtypes = []
    rt.cudaGetLastError.restype = ctypes.c_int
    rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcH), ctypes.c_void_p]
    rt.cudaIpcGetMemHandle.restype = ctypes.c_int
    rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcH, ctypes.c_uint]
    rt.cudaIpcOpenMemHandle.restype = ctypes.c_int
    rt.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    rt.cudaIpcCloseMemHandle.restype = ctypes.c_int
    return rt


def producer(q):
    rt = get_rt()
    rt.cudaSetDevice(0)
    ptr = ctypes.c_void_p()
    rt.cudaMalloc(ctypes.byref(ptr), 4096)
    rt.cudaMemset(ptr, 0x42, 16)
    rt.cudaDeviceSynchronize()
    h = IpcH()
    e = rt.cudaIpcGetMemHandle(ctypes.byref(h), ptr)
    q.put(bytes(h.reserved))
    print(f'P: err={e}, ptr=0x{ptr.value:x}', flush=True)
    time.sleep(8)
    rt.cudaFree(ptr)


def consumer(q):
    time.sleep(1)
    rt = get_rt()
    rt.cudaSetDevice(0)  # same device
    rt.cudaDeviceSynchronize()
    hb = q.get()
    h = IpcH()
    h.reserved = hb

    # Try flag=0
    p0 = ctypes.c_void_p()
    e0 = rt.cudaIpcOpenMemHandle(ctypes.byref(p0), h, 0)
    print(f'C flag=0: err={e0}', flush=True)

    # Try flag=1
    p1 = ctypes.c_void_p()
    e1 = rt.cudaIpcOpenMemHandle(ctypes.byref(p1), h, 1)
    print(f'C flag=1: err={e1}', flush=True)

    # Try without setting argtypes (let ctypes infer)
    rt2 = ctypes.CDLL('libcudart.so')
    p2 = ctypes.c_void_p()
    e2 = rt2.cudaIpcOpenMemHandle(ctypes.byref(p2), h, 0)
    print(f'C no-argtypes flag=0: err={e2}', flush=True)

    if p0.value: rt.cudaIpcCloseMemHandle(p0)
    if p1.value: rt.cudaIpcCloseMemHandle(p1)
    if p2.value: rt2.cudaIpcCloseMemHandle(p2)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=producer, args=(q, ))
    c = multiprocessing.Process(target=consumer, args=(q, ))
    p.start()
    c.start()
    p.join()
    c.join()
    print(f'exits: {p.exitcode}, {c.exitcode}')
