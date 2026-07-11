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

import ctypes
import sys
from typing import Tuple

import torch

TENSORMAP_SIZE_BYTES = 128

# --- Host tensormap (cuTensorMapEncodeTiled) for debug / comparison tests ---

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_void_p
cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64


class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]


CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 0x07
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 0x09
CU_TENSOR_MAP_INTERLEAVE_NONE = 0x00
CU_TENSOR_MAP_SWIZZLE_NONE = 0x00
CU_TENSOR_MAP_SWIZZLE_128B = 0x03
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 0x03
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0x00

_libcuda = None


def _cuda_driver():
    global _libcuda
    if _libcuda is None:
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        _libcuda = ctypes.CDLL(name)
        _libcuda.cuTensorMapEncodeTiled.argtypes = [
            ctypes.POINTER(CUtensorMap),
            ctypes.c_int,
            ctypes.c_uint32,
            CUdeviceptr,
            ctypes.POINTER(cuuint64_t),
            ctypes.POINTER(cuuint64_t),
            ctypes.POINTER(cuuint32_t),
            ctypes.POINTER(cuuint32_t),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _libcuda.cuTensorMapEncodeTiled.restype = CUresult
    return _libcuda


def _tensor_map_dtype(t: torch.Tensor) -> int:
    if t.dtype == torch.bfloat16:
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    if t.dtype == torch.float32:
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    raise TypeError(f"create_tensormap_2d: unsupported dtype {t.dtype}")


def create_tensormap_2d(
    tensor: torch.Tensor,
    tile_shape: Tuple[int, int],
    *,
    swizzle_bytes: int = 0,
) -> torch.Tensor:
    """Build a 128-byte TMA tensormap on the host via ``cuTensorMapEncodeTiled``.

    ``tensor`` must be 2-D row-major contiguous on CUDA. ``tile_shape`` is
    ``(BM, BN)`` box rows × cols (elements). Used by ``test_tmap_*`` comparison scripts.

    For device-side descriptor scratch, use :func:`create_tmap_scratch` instead.
    """
    if tensor.dim() != 2 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("create_tensormap_2d: expect 2-D contiguous CUDA tensor")
    BM, BN = int(tile_shape[0]), int(tile_shape[1])
    M, N = int(tensor.shape[0]), int(tensor.shape[1])
    elem = tensor.element_size()
    sw = int(swizzle_bytes)
    if sw not in (0, 128):
        raise ValueError("create_tensormap_2d: swizzle_bytes must be 0 or 128")
    smem_inner = BN if sw == 0 else sw // elem
    smem_outer = BM
    lib = _cuda_driver()
    tm = CUtensorMap()
    gdim = (cuuint64_t * 2)(N, M)
    sdim = (cuuint32_t * 2)(smem_inner, smem_outer)
    gstride = (cuuint64_t * 1)(N * elem)
    estr = (cuuint32_t * 2)(1, 1)
    swizzle = CU_TENSOR_MAP_SWIZZLE_NONE if sw == 0 else CU_TENSOR_MAP_SWIZZLE_128B
    err = lib.cuTensorMapEncodeTiled(
        ctypes.byref(tm),
        _tensor_map_dtype(tensor),
        2,
        CUdeviceptr(tensor.data_ptr()),
        gdim,
        gstride,
        sdim,
        estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if err != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with error code {err}")
    raw = bytes(tm.data)
    return torch.tensor(tuple(raw), dtype=torch.uint8, device=tensor.device)


def create_tmap_scratch(num_ctas: int, num_descs_per_cta: int = 1) -> torch.Tensor:
    """Allocate aligned global memory for device-side tensormap creation.

    Each tensormap needs 128 bytes in global memory. The storage must be
    128-byte aligned (PyTorch guarantees 512-byte alignment for CUDA tensors).

    Args:
        num_ctas: Number of CTAs (grid size), must be > 0
        num_descs_per_cta: Number of tensormaps per CTA (e.g., 2 for input+output), must be > 0

    Returns:
        torch.Tensor of uint8 on CUDA device
    """
    if num_ctas <= 0:
        raise ValueError(f"create_tmap_scratch: num_ctas must be positive, got {num_ctas}")
    if num_descs_per_cta <= 0:
        raise ValueError(f"create_tmap_scratch: num_descs_per_cta must be positive, got {num_descs_per_cta}")
    total_bytes = num_ctas * num_descs_per_cta * TENSORMAP_SIZE_BYTES
    return torch.zeros(total_bytes, dtype=torch.uint8, device="cuda")


__all__ = [
    "create_tmap_scratch",
    "create_tensormap_2d",
    "TENSORMAP_SIZE_BYTES",
]
