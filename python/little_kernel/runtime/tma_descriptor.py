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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# CUDA Driver API types
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_void_p
cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64


# CUtensorMap is 128 bytes
class CUtensorMap(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 128)]


# CUDA Tensor Map enums (CUtensorMapDataType)
CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0x00  # 1 byte
CU_TENSOR_MAP_DATA_TYPE_UINT16 = 0x01  # 2 bytes
CU_TENSOR_MAP_DATA_TYPE_UINT32 = 0x02  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_INT32 = 0x03  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_UINT64 = 0x04  # 8 bytes
CU_TENSOR_MAP_DATA_TYPE_INT64 = 0x05  # 8 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 0x06  # 2 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 0x07  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 0x08  # 8 bytes
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 0x09  # 2 bytes
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 0x0a  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 0x0b  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 0x0c  # 4 bytes
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B = 0x0d  # 4 bits
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 0x0e  # 4 bits
CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B = 0x0f  # 6 bits

CU_TENSOR_MAP_INTERLEAVE_NONE = 0x00
CU_TENSOR_MAP_INTERLEAVE_16B = 0x01
CU_TENSOR_MAP_INTERLEAVE_32B = 0x02

CU_TENSOR_MAP_SWIZZLE_NONE = 0x00
CU_TENSOR_MAP_SWIZZLE_32B = 0x01
CU_TENSOR_MAP_SWIZZLE_64B = 0x02
CU_TENSOR_MAP_SWIZZLE_128B = 0x03
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B = 0x04
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B = 0x05
CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B = 0x06

CU_TENSOR_MAP_L2_PROMOTION_NONE = 0x00
CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 0x01
CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 0x02
CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 0x03

CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0x00
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = 0x01


class TMADescriptorHelper:
    """Helper class to create TMA descriptors."""

    def __init__(self):
        self._libcuda = None
        self._load_library()

    def _load_library(self):
        """Load libcuda.so library."""
        if sys.platform == "win32":
            lib_name = "nvcuda.dll"
        else:
            lib_name = "libcuda.so.1"

        try:
            self._libcuda = ctypes.CDLL(lib_name)
        except OSError as e:
            raise RuntimeError(f"Failed to load {lib_name}: {e}")

        # Define cuTensorMapEncodeTiled signature
        self._libcuda.cuTensorMapEncodeTiled.argtypes = [
            ctypes.POINTER(CUtensorMap),  # tensorMap
            ctypes.c_int,  # tensorDataType
            ctypes.c_uint32,  # tensorRank
            CUdeviceptr,  # globalAddress
            ctypes.POINTER(cuuint64_t),  # globalDim
            ctypes.POINTER(cuuint64_t),  # globalStrides
            ctypes.POINTER(cuuint32_t),  # boxDim
            ctypes.POINTER(cuuint32_t),  # elementStrides
            ctypes.c_int,  # interleave
            ctypes.c_int,  # swizzle
            ctypes.c_int,  # l2Promotion
            ctypes.c_int,  # oobFill
        ]
        self._libcuda.cuTensorMapEncodeTiled.restype = CUresult

        self._libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        self._libcuda.cuGetErrorString.restype = CUresult

    def _check_error(self, err: int, func_name: str):
        """Check CUDA error and raise exception if needed."""
        if err != 0:  # CUDA_SUCCESS = 0
            error_str = ctypes.c_char_p()
            self._libcuda.cuGetErrorString(err, ctypes.byref(error_str))
            raise RuntimeError(
                f"CUDA error in {func_name}: {error_str.value.decode() if error_str.value else 'Unknown error'}")

    def _get_dtype(self, tensor) -> int:
        """Convert tensor dtype to CUDA tensor map dtype."""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            dtype = tensor.dtype
            if dtype == torch.bfloat16:
                return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            elif dtype == torch.float16:
                return CU_TENSOR_MAP_DATA_TYPE_FLOAT16
            elif dtype == torch.float32:
                return CU_TENSOR_MAP_DATA_TYPE_FLOAT32
            elif dtype == torch.int32:
                return CU_TENSOR_MAP_DATA_TYPE_INT32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        else:
            raise ValueError("Only PyTorch tensors are supported")

    def _get_swizzle_mode(self, swizzle_mode: int, swizzle_base: int = 0) -> int:
        """Convert swizzle mode to CUDA enum."""
        if swizzle_base != 0:
            if swizzle_base == 32 and swizzle_mode == 128:
                return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
            raise ValueError(f"Unsupported swizzle_base: {swizzle_base}")

        if swizzle_mode == 0 or swizzle_mode == 16:
            return CU_TENSOR_MAP_SWIZZLE_NONE
        elif swizzle_mode == 32:
            return CU_TENSOR_MAP_SWIZZLE_32B
        elif swizzle_mode == 64:
            return CU_TENSOR_MAP_SWIZZLE_64B
        elif swizzle_mode == 128:
            return CU_TENSOR_MAP_SWIZZLE_128B
        else:
            raise ValueError(f"Unsupported swizzle_mode: {swizzle_mode}")

    def create_2d_descriptor(
        self,
        tensor,
        gmem_inner_dim: int,
        gmem_outer_dim: int,
        smem_inner_dim: int,
        smem_outer_dim: int,
        gmem_outer_stride: int,
        swizzle_mode: int = 0,
        swizzle_base: int = 0,
        oob_fill: bool = False,
        l2_promotion: int = CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    ) -> CUtensorMap:
        """
        Create a 2D TMA descriptor.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor
        gmem_inner_dim : int
            Global memory inner dimension (number of elements in the
            contiguous / innermost dimension of global memory).
        gmem_outer_dim : int
            Global memory outer dimension (number of rows / outer dim).
        smem_inner_dim : int
            Shared memory box inner dimension (elements per row to copy).
            When *swizzle_mode* != 0 this is automatically clamped to
            ``swizzle_mode // element_size``.
        smem_outer_dim : int
            Shared memory box outer dimension (rows to copy).
        gmem_outer_stride : int
            Global memory outer stride **in elements** (not bytes).
            Typically ``tensor.stride(0)`` for a row-major 2-D tensor.
        swizzle_mode : int
            Swizzle mode in bytes: 0 (none), 32, 64, or 128.
        swizzle_base : int
            Swizzle base (default 0).
        oob_fill : bool
            If True, out-of-bounds accesses return zero (useful for TMA
            loads).  If False, no OOB filling (required for TMA stores).
            Default is False.
        l2_promotion : int
            L2 promotion hint.  Default ``CU_TENSOR_MAP_L2_PROMOTION_L2_256B``.
        
        Returns
        -------
        CUtensorMap
            TMA descriptor ready to pass to a kernel.
        """
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            raise ValueError("tensor must be a PyTorch tensor")

        elem_size = tensor.element_size()
        if swizzle_mode != 0:
            smem_inner_dim = swizzle_mode // elem_size

        tensor_map = CUtensorMap()
        gmem_dims = (cuuint64_t * 2)(gmem_inner_dim, gmem_outer_dim)
        smem_dims = (cuuint32_t * 2)(smem_inner_dim, smem_outer_dim)
        gmem_strides = (cuuint64_t * 1)(gmem_outer_stride * elem_size)
        elem_strides = (cuuint32_t * 2)(1, 1)

        oob_fill_enum = (CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                         if oob_fill else CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)

        err = self._libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tensor_map),
            self._get_dtype(tensor),
            2,  # tensorRank
            CUdeviceptr(tensor.data_ptr()),
            gmem_dims,
            gmem_strides,
            smem_dims,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            self._get_swizzle_mode(swizzle_mode, swizzle_base),
            l2_promotion,
            oob_fill_enum,
        )
        self._check_error(err, "cuTensorMapEncodeTiled")

        return tensor_map


# Global helper instance
_tma_helper = None


def get_tma_helper() -> TMADescriptorHelper:
    """Get global TMA descriptor helper instance."""
    global _tma_helper
    if _tma_helper is None:
        _tma_helper = TMADescriptorHelper()
    return _tma_helper


def create_tma_2d_descriptor(
    tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int = 0,
    swizzle_base: int = 0,
    oob_fill: bool = False,
    l2_promotion: int = CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
) -> CUtensorMap:
    """Create a 2D TMA descriptor (convenience function).

    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor on CUDA device.
    gmem_inner_dim, gmem_outer_dim : int
        Global memory dimensions (elements).
    smem_inner_dim, smem_outer_dim : int
        Shared-memory box dimensions (elements).  *smem_inner_dim* is
        auto-adjusted when *swizzle_mode* != 0.
    gmem_outer_stride : int
        Outer stride **in elements** (e.g. ``tensor.stride(0)``).
    swizzle_mode : int
        Swizzle in bytes: 0, 32, 64, or 128.
    swizzle_base : int
        Swizzle base (default 0).
    oob_fill : bool
        Zero-fill out-of-bound accesses (True for loads, False for stores).
    l2_promotion : int
        L2 promotion hint (default 256B).

    Returns
    -------
    CUtensorMap
        Descriptor to pass as ``grid_constant[TmaDescriptor]`` kernel arg.
    """
    return get_tma_helper().create_2d_descriptor(
        tensor,
        gmem_inner_dim,
        gmem_outer_dim,
        smem_inner_dim,
        smem_outer_dim,
        gmem_outer_stride,
        swizzle_mode,
        swizzle_base,
        oob_fill,
        l2_promotion,
    )
