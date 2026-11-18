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
from dataclasses import dataclass
from little_kernel.backends.nv.common.runtime._tensor_map import CUtensorMap

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

# The CUDA Driver requires a CUtensorMap passed to cuTensorMapEncode*/
# cuTensorMapReplaceAddress to be 64-byte aligned.  ctypes does not honor an
# ``_align_`` declaration, so aligned storage is allocated manually: a raw
# buffer with 63 padding bytes, an aligned offset found at runtime, and the
# CUtensorMap view obtained by pointer cast.  The underlying buffer is
# retained on the object (``_lk_storage``) for the lifetime of the
# descriptor, which must also outlive any driver call.
_TENSOR_MAP_ALIGNMENT = 64
_TENSOR_MAP_SIZE = 128


def _new_aligned_tensor_map() -> CUtensorMap:
    """Return zeroed, 64-byte-aligned CUtensorMap storage.

    Returns a pointer-cast view whose own address is 64-byte aligned; the raw
    allocation is kept alive on the object (``_lk_storage``) so the Driver
    never sees freed storage.
    """
    pad = _TENSOR_MAP_ALIGNMENT - 1
    storage = (ctypes.c_ubyte * (_TENSOR_MAP_SIZE + pad))()
    address = ctypes.addressof(storage)
    offset = (-address) % _TENSOR_MAP_ALIGNMENT
    view = ctypes.cast(ctypes.addressof(storage) + offset, ctypes.POINTER(CUtensorMap)).contents
    view._lk_storage = storage  # retain the allocation
    return view


# ---------------------------------------------------------------------------
# Byte-buffer transport
# ---------------------------------------------------------------------------
#
# The export path hands host-encoded descriptors between Python, generated
# bindings and staging code as an opaque byte buffer: one CUtensorMap is
# exactly ``_TENSOR_MAP_SIZE`` (128) bytes, and a per-batch bundle of N
# descriptors is the N concatenated payloads (``N * 128`` bytes, matching the
# C++ ``sizeof(CUtensorMap) * count`` device-array staging).  The payload
# bytes are Driver-owned and never interpreted here; the round trip only
# copies bytes in and out of correctly aligned storage.


def tensor_map_to_bytes(tensor_map: CUtensorMap) -> bytes:
    """Copy the opaque 128-byte Driver payload out of a host ``CUtensorMap``.

    The returned bytes are a snapshot: mutating or freeing the source
    descriptor afterwards does not affect them.  Bytes may cross process or
    serialization boundaries only as opaque Driver data; only
    :func:`tensor_map_from_bytes` turns them back into a Driver-ready,
    64-byte-aligned descriptor.
    """
    if not isinstance(tensor_map, CUtensorMap):
        raise TypeError(f"tensor_map must be CUtensorMap, got {type(tensor_map)!r}")
    return ctypes.string_at(ctypes.byref(tensor_map), _TENSOR_MAP_SIZE)


def tensor_maps_to_bytes(tensor_maps) -> bytes:
    """Concatenate a non-empty sequence of descriptors into one byte buffer.

    This is the per-batch bundle transport: the returned buffer has
    ``len(tensor_maps) * 128`` bytes with descriptor ``i`` at byte offset
    ``i * 128``, the same layout generated bindings upload to a device
    ``CUtensorMap`` array.
    """
    if isinstance(tensor_maps, (CUtensorMap, bytes, bytearray, memoryview, str)):
        raise TypeError("tensor_maps must be a sequence of CUtensorMap, got "
                        f"{type(tensor_maps)!r}")
    try:
        maps = tuple(tensor_maps)
    except TypeError:
        raise TypeError(f"tensor_maps must be a sequence of CUtensorMap, got "
                        f"{type(tensor_maps)!r}") from None
    if not maps:
        raise ValueError("tensor_maps must contain at least one descriptor")
    parts = []
    for index, tensor_map in enumerate(maps):
        if not isinstance(tensor_map, CUtensorMap):
            raise TypeError(f"tensor_maps[{index}] must be CUtensorMap, got "
                            f"{type(tensor_map)!r}")
        parts.append(ctypes.string_at(ctypes.byref(tensor_map), _TENSOR_MAP_SIZE))
    return b"".join(parts)


def tensor_map_from_bytes(payload) -> CUtensorMap:
    """Rebuild a 64-byte-aligned ``CUtensorMap`` from its 128-byte payload.

    ``payload`` must be a bytes-like object of exactly ``_TENSOR_MAP_SIZE``
    bytes, e.g. the output of :func:`tensor_map_to_bytes` or one 128-byte
    slice of a bundle buffer.  The result owns fresh, correctly aligned
    storage (the Driver requires 64-byte alignment for encode/replace calls),
    so it can be passed straight back to the Driver API; the bytes themselves
    are never validated or interpreted -- Driver validation happens at the
    next Driver call.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be a bytes-like object, got {type(payload)!r}")
    data = bytes(payload)
    if len(data) != _TENSOR_MAP_SIZE:
        raise ValueError(f"payload must be exactly {_TENSOR_MAP_SIZE} bytes, got "
                         f"{len(data)}")
    tensor_map = _new_aligned_tensor_map()
    ctypes.memmove(ctypes.byref(tensor_map), data, _TENSOR_MAP_SIZE)
    return tensor_map


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


@dataclass(frozen=True)
class TiledTensorMap2DPlan:
    """Host-side, pointer-explicit plan for a rank-2 tiled TMA descriptor.

    Unlike the PyTorch-tensor convenience wrapper, this plan accepts a runtime
    integer device address.  Grouped GEMMs can therefore build one plan per
    group by adding a host-visible group offset (for example an offset decoded
    from ``row_ptr`` metadata) to a common base, without tying descriptor
    construction to a particular frontend tensor object or task shape.
    """

    global_address: int
    global_inner_dim: int
    global_outer_dim: int
    box_inner_dim: int
    box_outer_dim: int
    outer_stride_bytes: int
    data_type: int
    element_size_bytes: int
    swizzle: int = CU_TENSOR_MAP_SWIZZLE_128B
    interleave: int = CU_TENSOR_MAP_INTERLEAVE_NONE
    l2_promotion: int = CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    oob_fill: int = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

    def __post_init__(self):
        values = {
            "global_inner_dim": self.global_inner_dim,
            "global_outer_dim": self.global_outer_dim,
            "box_inner_dim": self.box_inner_dim,
            "box_outer_dim": self.box_outer_dim,
            "outer_stride_bytes": self.outer_stride_bytes,
            "element_size_bytes": self.element_size_bytes,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not isinstance(self.global_address, int) or isinstance(self.global_address, bool):
            raise TypeError(f"global_address must be an integer, got {self.global_address!r}")
        if self.global_address < 0:
            raise ValueError("global_address must be a nonnegative device address")
        if self.global_address % 16 != 0:
            raise ValueError("TMA global_address must be at least 16-byte aligned")
        if self.global_address >= (1 << (8 * ctypes.sizeof(CUdeviceptr))):
            raise ValueError(f"global_address 0x{self.global_address:x} does not fit in a "
                             f"{ctypes.sizeof(CUdeviceptr) * 8}-bit device pointer")
        # The driver caps every box dimension at 256 elements. Boundary boxes
        # are allowed to exceed the global dimensions: TMA zero-fills (loads)
        # / ignores (stores) the out-of-bounds lanes, so do not reject them.
        for name, value in (("box_inner_dim", self.box_inner_dim), ("box_outer_dim", self.box_outer_dim)):
            if not 1 <= value <= 256:
                raise ValueError(f"{name} must be in 1..256 elements, got {value}")
        contiguous_stride = self.global_inner_dim * self.element_size_bytes
        if self.outer_stride_bytes < contiguous_stride:
            raise ValueError("outer_stride_bytes must cover one inner dimension: got "
                             f"{self.outer_stride_bytes}, need at least {contiguous_stride}")
        if self.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE:
            # No interleave: the inner box must be built from whole 16-byte
            # TMA atoms, and swizzled layouts additionally cap it at the
            # selected swizzle span. This is the driver rule (verified on
            # H800), not the old inverted ">= swizzle span in 128-byte
            # multiples" check: e.g. fp16 + swizzle128 accepts inner boxes of
            # 32 elements (64B) and 64 elements (128B), but rejects 128
            # elements (256B).
            box_bytes = self.box_inner_dim * self.element_size_bytes
            if box_bytes % 16 != 0:
                raise ValueError("TMA inner box must be a multiple of 16 bytes with no "
                                 f"interleave, got {box_bytes}")
            swizzle_span = {
                CU_TENSOR_MAP_SWIZZLE_32B: 32,
                CU_TENSOR_MAP_SWIZZLE_64B: 64,
                CU_TENSOR_MAP_SWIZZLE_128B: 128,
                # The atom-32/atom-64 variants still swizzle a 128-byte span;
                # the Driver decides whether a specific device supports them,
                # but the byte-span legality rule is architecture-independent.
                CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B: 128,
                CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B: 128,
                CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B: 128,
            }.get(self.swizzle)
            if swizzle_span is not None and box_bytes > swizzle_span:
                raise ValueError(f"TMA inner box of {box_bytes} bytes exceeds the selected "
                                 f"swizzle span of {swizzle_span} bytes")

    def with_address(self, address: int) -> "TiledTensorMap2DPlan":
        """Return an identical descriptor plan rebound to another device base."""
        if not isinstance(address, int) or isinstance(address, bool):
            raise TypeError(f"address must be an integer, got {address!r}")
        return TiledTensorMap2DPlan(
            global_address=address,
            global_inner_dim=self.global_inner_dim,
            global_outer_dim=self.global_outer_dim,
            box_inner_dim=self.box_inner_dim,
            box_outer_dim=self.box_outer_dim,
            outer_stride_bytes=self.outer_stride_bytes,
            data_type=self.data_type,
            element_size_bytes=self.element_size_bytes,
            swizzle=self.swizzle,
            interleave=self.interleave,
            l2_promotion=self.l2_promotion,
            oob_fill=self.oob_fill,
        )


def build_grouped_tiled_2d_plans(
    base_address: int,
    group_byte_offsets,
    template: TiledTensorMap2DPlan,
):
    """Rebase one tiled 2-D plan for each runtime group byte offset.

    ``group_byte_offsets`` is intentionally a generic sequence: callers may
    derive it from row-pointer metadata, group strides, or any other
    host-visible schedule.  The shape/box/swizzle fields are shared; only the
    TMA global address changes.
    """
    if not isinstance(base_address, int) or isinstance(base_address, bool):
        raise TypeError(f"base_address must be an integer, got {base_address!r}")
    offsets = tuple(group_byte_offsets)
    if not offsets:
        raise ValueError("group_byte_offsets must contain at least one group")
    plans = []
    for offset in offsets:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"group byte offsets must be nonnegative integers, got {offset!r}")
        plans.append(template.with_address(base_address + offset))
    return tuple(plans)


def _validate_device_address(global_address: int) -> None:
    """Validate an integer device address before touching the CUDA Driver.

    Rejects booleans, non-integers, negatives, non-16-byte-aligned values and
    values that cannot fit a device pointer (so the Driver is never loaded
    for an obviously invalid address).
    """
    if not isinstance(global_address, int) or isinstance(global_address, bool):
        raise TypeError(f"global_address must be an integer device address, got {global_address!r}")
    if global_address < 0:
        raise ValueError("global_address must be a nonnegative device address")
    if global_address % 16 != 0:
        raise ValueError("global_address must be 16-byte aligned for a TMA descriptor")
    pointer_bits = ctypes.sizeof(CUdeviceptr) * 8
    if global_address >= (1 << pointer_bits):
        raise ValueError(f"global_address 0x{global_address:x} does not fit in a "
                         f"{pointer_bits}-bit device pointer")


def rebased_cu_tensor_map(template: CUtensorMap, global_address: int) -> CUtensorMap:
    """Copy a host ``CUtensorMap`` and rebind its global address via the Driver.

    The opaque CUtensorMap payload is never interpreted: the CUDA Driver API
    ``cuTensorMapReplaceAddress`` is the only supported way to change the
    embedded global address (it is not a fixed-offset first field across CUDA
    versions). Validation happens before the Driver is loaded; the call runs
    on a 64-byte-aligned copy (the Driver requires 64-byte alignment and the
    storage must outlive the call), so the template is preserved byte-for-byte
    on both success and error paths.
    """
    if not isinstance(template, CUtensorMap):
        raise TypeError(f"template must be CUtensorMap, got {type(template)!r}")
    _validate_device_address(global_address)

    result = _new_aligned_tensor_map()
    ctypes.memmove(ctypes.byref(result), ctypes.byref(template), ctypes.sizeof(CUtensorMap))

    helper = get_tma_helper()
    replace_address = helper._libcuda.cuTensorMapReplaceAddress
    if not hasattr(replace_address, "_lk_signature_set"):
        replace_address.argtypes = [
            ctypes.POINTER(CUtensorMap),  # tensorMap
            CUdeviceptr,  # globalAddress
        ]
        replace_address.restype = CUresult
        replace_address._lk_signature_set = True
    err = replace_address(ctypes.byref(result), CUdeviceptr(global_address))
    # _check_error raises on failure; the template was never passed to the
    # Driver and is therefore preserved on both success and error paths.
    helper._check_error(err, "cuTensorMapReplaceAddress")
    return result


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

        # cuTensorMapReplaceAddress rebinds the opaque global address inside
        # an existing, 64-byte-aligned CUtensorMap without re-encoding it.
        self._libcuda.cuTensorMapReplaceAddress.argtypes = [
            ctypes.POINTER(CUtensorMap),  # tensorMap
            CUdeviceptr,  # globalAddress
        ]
        self._libcuda.cuTensorMapReplaceAddress.restype = CUresult

        self._libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        self._libcuda.cuGetErrorString.restype = CUresult

    def _check_error(self, err: int, func_name: str):
        """Check CUDA error and raise exception if needed."""
        if err != 0:  # CUDA_SUCCESS = 0
            error_str = ctypes.c_char_p()
            self._libcuda.cuGetErrorString(err, ctypes.byref(error_str))
            raise RuntimeError(
                f"CUDA error in {func_name}: {error_str.value.decode() if error_str.value else 'Unknown error'}")

    def encode_tiled_2d_plan(self, plan: TiledTensorMap2DPlan) -> CUtensorMap:
        """Call ``cuTensorMapEncodeTiled`` for a pointer-explicit 2-D plan."""
        if not isinstance(plan, TiledTensorMap2DPlan):
            raise TypeError(f"plan must be TiledTensorMap2DPlan, got {type(plan)!r}")
        tensor_map = _new_aligned_tensor_map()
        gmem_dims = (cuuint64_t * 2)(plan.global_inner_dim, plan.global_outer_dim)
        gmem_strides = (cuuint64_t * 1)(plan.outer_stride_bytes)
        smem_dims = (cuuint32_t * 2)(plan.box_inner_dim, plan.box_outer_dim)
        elem_strides = (cuuint32_t * 2)(1, 1)
        err = self._libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tensor_map),
            plan.data_type,
            2,
            CUdeviceptr(plan.global_address),
            gmem_dims,
            gmem_strides,
            smem_dims,
            elem_strides,
            plan.interleave,
            plan.swizzle,
            plan.l2_promotion,
            plan.oob_fill,
        )
        self._check_error(err, "cuTensorMapEncodeTiled")
        return tensor_map

    def encode_grouped_tiled_2d(
        self,
        base_address: int,
        group_byte_offsets,
        template: TiledTensorMap2DPlan,
    ):
        """Encode one host descriptor per rebased grouped 2-D plan."""
        plans = build_grouped_tiled_2d_plans(base_address, group_byte_offsets, template)
        return tuple(self.encode_tiled_2d_plan(plan) for plan in plans)

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
            elif dtype == torch.uint8:
                return CU_TENSOR_MAP_DATA_TYPE_UINT8
            elif dtype == torch.int8:
                return CU_TENSOR_MAP_DATA_TYPE_UINT8  # CUDA maps INT8 to UINT8 for TMA
            elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                return CU_TENSOR_MAP_DATA_TYPE_UINT8  # TMA treats 8-bit floats as raw bytes
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

        tensor_map = _new_aligned_tensor_map()
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

    def create_3d_descriptor(
        self,
        tensor,
        dim0: int,
        dim1: int,
        dim2: int,
        stride1_bytes: int,
        stride2_bytes: int,
        tile_dim0: int,
        tile_dim1: int,
        swizzle_mode: int = 128,
        oob_fill: bool = False,
        l2_promotion: int = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        tensor_map_dtype: int | None = None,
    ) -> CUtensorMap:
        """
        Create a 3D TMA descriptor.

        Layout: [dim0, dim1, dim2] with dim0 innermost (e.g. headdim, seqlen, heads).
        tensor_map_dtype overrides dtype inference for packed TMA formats such as 16U4_ALIGN16B.
        """
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            raise ValueError("tensor must be a PyTorch tensor")
        tensor_map = _new_aligned_tensor_map()
        gmem_dims = (cuuint64_t * 3)(dim0, dim1, dim2)
        gmem_strides = (cuuint64_t * 2)(stride1_bytes, stride2_bytes)
        box_dim = (cuuint32_t * 3)(tile_dim0, tile_dim1, 1)
        elem_strides = (cuuint32_t * 3)(1, 1, 1)
        oob_fill_enum = (CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                         if oob_fill else CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        err = self._libcuda.cuTensorMapEncodeTiled(
            ctypes.byref(tensor_map),
            self._get_dtype(tensor) if tensor_map_dtype is None else tensor_map_dtype,
            3,
            CUdeviceptr(tensor.data_ptr()),
            gmem_dims,
            gmem_strides,
            box_dim,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            self._get_swizzle_mode(swizzle_mode, 0),
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


def create_grouped_tma_2d_descriptors(
    base_address: int,
    group_byte_offsets,
    template: TiledTensorMap2DPlan,
):
    """Encode grouped host descriptors from runtime group byte offsets."""
    return get_tma_helper().encode_grouped_tiled_2d(base_address, group_byte_offsets, template)


def create_tma_3d_descriptor(
    tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    stride1_bytes: int,
    stride2_bytes: int,
    tile_dim0: int,
    tile_dim1: int,
    swizzle_mode: int = 128,
    oob_fill: bool = False,
    l2_promotion: int = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    tensor_map_dtype: int | None = None,
) -> CUtensorMap:
    """Create a 3D TMA descriptor, optionally overriding the CUDA tensor-map dtype."""
    return get_tma_helper().create_3d_descriptor(
        tensor,
        dim0,
        dim1,
        dim2,
        stride1_bytes,
        stride2_bytes,
        tile_dim0,
        tile_dim1,
        swizzle_mode,
        oob_fill,
        l2_promotion,
        tensor_map_dtype,
    )
