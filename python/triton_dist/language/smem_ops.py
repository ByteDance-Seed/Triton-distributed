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

import math
import triton.language as tl
from triton.language import core as tlc
from triton.language.core import builtin, constexpr

# MLIR/NVVM generic AS for pointers from shared memdesc (matches LLVM ptr addrspace(0)).
SMEM_GENERIC_POINTER_ADDR_SPACE = 0


def _smem_generic_ptr_ty(element_ty):
    return tlc.pointer_type(element_ty, address_space=SMEM_GENERIC_POINTER_ADDR_SPACE)


class SharedMemoryDesc(tlc.tensor):
    """Python wrapper around a memdesc IR handle for shared memory.

    Inherits from :class:`tlc.tensor`.  ``self.type`` is :class:`SharedMemDescType`
    so nested ``@jit`` uses the same memdesc MLIR type for ``tt.call`` as for
    ``local_alloc`` (analogous to :class:`tlc.block_type` for block tensors).
    """
    __triton_builtin__ = True

    def __init__(
        self,
        handle,
        element_ty,
        shape,
        alloc_shape=None,
        vec: int = 1,
        per_phase: int = 1,
        max_phase: int = 1,
    ):
        if alloc_shape is None:
            alloc_shape = list(shape)
        else:
            alloc_shape = list(alloc_shape)
        self.handle = handle
        self.element_ty = element_ty
        self.smem_shape = list(shape)
        self.alloc_shape = alloc_shape
        self.type = SharedMemDescType(
            element_ty,
            tuple(shape),
            tuple(alloc_shape),
            int(vec),
            int(per_phase),
            int(max_phase),
        )
        self.dtype = element_ty
        self.shape = tuple(constexpr(s) for s in shape)
        self.numel = constexpr(math.prod(shape))

    def _flatten_ir(self, handles):
        handles.append(self.handle)

    @property
    def rank(self):
        return len(self.smem_shape)

    @property
    def nbytes(self):
        return math.prod(self.smem_shape) * (self.element_ty.primitive_bitwidth // 8)


class SharedMemDescType(tlc.base_type):
    """Frontend type for ``ttg`` shared memdesc (matches ``local_alloc`` / subview)."""

    __slots__ = ("element_ty", "shape", "alloc_shape", "vec", "per_phase", "max_phase")

    def __init__(
        self,
        element_ty,
        shape: tuple,
        alloc_shape: tuple,
        vec: int,
        per_phase: int,
        max_phase: int,
    ):
        self.element_ty = element_ty
        self.shape = tuple(shape)
        self.alloc_shape = tuple(alloc_shape)
        self.vec = int(vec)
        self.per_phase = int(per_phase)
        self.max_phase = int(max_phase)

    @staticmethod
    def is_block() -> bool:
        return False

    @staticmethod
    def is_ptr() -> bool:
        return False

    @property
    def scalar(self):
        return self.element_ty

    def __eq__(self, other):
        if not isinstance(other, SharedMemDescType):
            return False
        return (self.element_ty == other.element_ty and self.shape == other.shape
                and self.alloc_shape == other.alloc_shape and self.vec == other.vec
                and self.per_phase == other.per_phase and self.max_phase == other.max_phase)

    def __hash__(self):
        return hash((
            self.element_ty,
            self.shape,
            self.alloc_shape,
            self.vec,
            self.per_phase,
            self.max_phase,
        ))

    def __repr__(self):
        return (f"SharedMemDescType(element_ty={self.element_ty}, shape={self.shape}, "
                f"alloc_shape={self.alloc_shape}, vec={self.vec}, "
                f"per_phase={self.per_phase}, max_phase={self.max_phase})")

    def mangle(self) -> str:
        elt = self.element_ty.mangle()
        sh = "_".join(str(s) for s in self.shape)
        ah = "_".join(str(s) for s in self.alloc_shape)
        return f"M{elt}S{sh}A{ah}V{self.vec}_{self.per_phase}_{self.max_phase}M"

    def _flatten_ir_types(self, builder, out):
        rank = len(self.shape)
        order = list(range(rank - 1, -1, -1))
        elem_ir_ty = self.element_ty.to_ir(builder)
        layout = builder.get_swizzled_shared_layout(self.vec, self.per_phase, self.max_phase, order)
        memdesc_ty = builder.get_shared_mem_desc_ty(elem_ir_ty, list(self.shape), layout, list(self.alloc_shape))
        out.append(memdesc_ty)

    def _unflatten_ir(self, handles, cursor):
        v = SharedMemoryDesc(
            handles[cursor],
            self.element_ty,
            list(self.shape),
            list(self.alloc_shape),
            self.vec,
            self.per_phase,
            self.max_phase,
        )
        return v, cursor + 1


def _to_ir_index(val, semantic):
    if isinstance(val, constexpr):
        return semantic._convert_elem_to_ir_value(val, require_i64=False)
    elif isinstance(val, tlc.tensor):
        return val.handle
    elif isinstance(val, int):
        return semantic._convert_elem_to_ir_value(val, require_i64=False)
    return val


def _to_ir_stored_scalar(val, semantic):
    """IR handle for a value stored to shared mem (tensor scalar or trace-time constant)."""
    if isinstance(val, tlc.tensor):
        return val.handle
    if isinstance(val, constexpr):
        if isinstance(val.value, bool):
            return semantic._convert_elem_to_ir_value(int(val.value), require_i64=False)
        return semantic._convert_elem_to_ir_value(val, require_i64=False)
    if isinstance(val, bool):
        return semantic._convert_elem_to_ir_value(int(val), require_i64=False)
    if isinstance(val, int):
        return semantic._convert_elem_to_ir_value(val, require_i64=False)
    return val


def _memdesc_handle_and_element_ty(smem, op_name: str):
    """Resolve memdesc handle + element type for allocate_smem results.

    Nested jit callees may receive a proxy that is not a SharedMemoryDesc
    instance but still exposes handle and element_ty.
    """
    if isinstance(smem, SharedMemoryDesc):
        return smem.handle, smem.element_ty
    handle = getattr(smem, "handle", None)
    element_ty = getattr(smem, "element_ty", None)
    if handle is not None and element_ty is not None:
        return handle, element_ty
    raise TypeError(f"{op_name}: expected shared memdesc (SharedMemoryDesc or handle+element_ty)")


@builtin
def allocate_smem(element_ty, shape, vec: constexpr = constexpr(1), per_phase: constexpr = constexpr(1),
                  max_phase: constexpr = constexpr(1), _semantic=None):
    """Allocate shared memory with a swizzled layout.

    Goes through the compiler IR (ttg.local_alloc) for safe allocation.

    Args:
        element_ty: Element data type (e.g., tl.bfloat16, tl.int64).
        shape: List of dimension sizes.
        vec, per_phase, max_phase: Swizzled layout parameters.

    Returns:
        SharedMemoryDesc wrapping the compiler-allocated memdesc.
    """
    if isinstance(element_ty, constexpr):
        element_ty = element_ty.value
    if isinstance(shape, constexpr):
        shape = shape.value
    shape = [s.value if isinstance(s, constexpr) else s for s in shape]
    if not shape:
        raise ValueError("allocate_smem: shape must be non-empty")
    if any(s <= 0 for s in shape):
        raise ValueError(f"allocate_smem: all dimensions must be positive, got {shape}")
    vec_val = vec.value if isinstance(vec, constexpr) else vec
    pp_val = per_phase.value if isinstance(per_phase, constexpr) else per_phase
    mp_val = max_phase.value if isinstance(max_phase, constexpr) else max_phase

    rank = len(shape)
    order = list(range(rank - 1, -1, -1))

    builder = _semantic.builder
    elem_ir_ty = element_ty.to_ir(builder)
    layout = builder.get_swizzled_shared_layout(vec_val, pp_val, mp_val, order)
    alloc_shape = list(shape)
    memdesc_ty = builder.get_shared_mem_desc_ty(elem_ir_ty, shape, layout, alloc_shape)
    handle = builder.create_local_alloc(memdesc_ty)
    return SharedMemoryDesc(
        handle,
        element_ty,
        shape,
        alloc_shape,
        vec_val,
        pp_val,
        mp_val,
    )


@builtin
def smem_index(smem, index, _semantic=None):
    """Index into shared memory along dimension 0 (subview).

    Returns a SharedMemoryDesc with rank-1 dimensions.

    Requires a full :class:`SharedMemoryDesc` (shape metadata), not a bare
    memdesc proxy from nested ``@jit`` callees.

    .. note::

       The subview is created with swizzle ``(vec=1, per_phase=1, max_phase=1)``
       regardless of the parent descriptor's swizzle.  This is correct for the
       common pipeline pattern (``allocate_smem(..., [NUM_STAGES, BM, BN])`` with
       default swizzle, then ``smem_index(buf, stage_idx)``).  If the parent uses
       non-trivial swizzle and the subview's consumer relies on matching swizzle
       metadata, propagate the parent's params explicitly or use a dedicated
       subview helper.
    """
    assert isinstance(smem, SharedMemoryDesc), "smem must be a SharedMemoryDesc"
    assert smem.rank > 1, "Cannot index a 1D shared memory descriptor"

    builder = _semantic.builder
    idx_val = _to_ir_index(index, _semantic)

    sub_shape = smem.smem_shape[1:]
    sub_alloc_shape = sub_shape
    elem_ir_ty = smem.element_ty.to_ir(builder)
    rank = len(sub_shape)
    order = list(range(rank - 1, -1, -1))
    layout = builder.get_swizzled_shared_layout(1, 1, 1, order)
    sub_memdesc_ty = builder.get_shared_mem_desc_ty(elem_ir_ty, sub_shape, layout, sub_alloc_shape)

    offsets = [idx_val] + [builder.get_int32(0)] * (len(smem.smem_shape) - 1)
    handle = builder.create_memdesc_subview(sub_memdesc_ty, smem.handle, offsets)
    return SharedMemoryDesc(
        handle,
        smem.element_ty,
        sub_shape,
        sub_alloc_shape,
        1,
        1,
        1,
    )


@builtin
def smem_load(smem, indices, _semantic=None):
    """Load a scalar element from shared memory at given indices."""
    handle, element_ty = _memdesc_handle_and_element_ty(smem, "smem_load")
    idx_handles = [_to_ir_index(i, _semantic) for i in indices]
    result = _semantic.builder.create_load_shared(handle, idx_handles)
    return tlc.tensor(result, element_ty)


@builtin
def smem_store(smem, value, indices, _semantic=None):
    """Store a scalar value to shared memory at given indices.

    ``value`` must be a :class:`triton.language.core.tensor` scalar, or a
    trace-time constant (``constexpr`` / ``int`` / ``bool``) compatible with the
    memdesc element type. For typed literals (e.g. bf16), use ``tl.cast``.
    """
    handle, _ = _memdesc_handle_and_element_ty(smem, "smem_store")
    val_handle = _to_ir_stored_scalar(value, _semantic)
    idx_handles = [_to_ir_index(i, _semantic) for i in indices]
    _semantic.builder.create_store_shared(val_handle, handle, idx_handles)


@builtin
def smem_get_ptr(smem, indices=None, _semantic=None):
    """Get a raw generic-address-space pointer to a shared memory location.

    The returned pointer can be passed to inline PTX for mbarrier, TMA, etc.
    """
    handle, element_ty = _memdesc_handle_and_element_ty(smem, "smem_get_ptr")
    builder = _semantic.builder
    idx_handles = [_to_ir_index(i, _semantic) for i in (indices or [])]
    ptr_ty = _smem_generic_ptr_ty(element_ty)
    result = builder.create_memdesc_to_ptr(handle, idx_handles, ptr_ty.to_ir(builder))
    return tlc.tensor(result, ptr_ty)


@builtin
def get_smem_shared_address_u32(smem, indices=None, _semantic=None):
    """Shared-memory offset as ``uint32`` for PTX operands (``.shared::cta``).

    Same pattern as CUTLASS/CUTE ``cast_smem_ptr_to_uint``: ``cvta.to.shared``
    then narrow to 32 bits. Combines ``memdesc_to_ptr`` with the same PTX as
    ``tma_language._ptr_to_shared_u32`` (which takes a pre-built generic ptr).
    The duplication is intentional to avoid coupling these two modules.

    """
    handle, element_ty = _memdesc_handle_and_element_ty(smem, "get_smem_shared_address_u32")
    builder = _semantic.builder
    idx_handles = [_to_ir_index(i, _semantic) for i in (indices or [])]
    ptr_ty = _smem_generic_ptr_ty(element_ty)
    ptr_handle = builder.create_memdesc_to_ptr(handle, idx_handles, ptr_ty.to_ir(builder))
    generic_ptr = tlc.tensor(ptr_handle, ptr_ty)
    ptr_u64 = tl.cast(generic_ptr, tl.uint64, bitcast=True, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u64 %generic_addr;
            mov.u64 %generic_addr, $1;
            cvta.to.shared.u64 %generic_addr, %generic_addr;
            cvt.u32.u64 $0, %generic_addr;
        }
        """,
        constraints="=r,l",
        args=[ptr_u64],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def smem_dealloc(smem, _semantic=None):
    """End the dynamic shared allocation lifetime (``ttg.local_dealloc``).

    Call once per allocation after the last use; all threads in the kernel
    should execute the same control flow leading to dealloc (same as other
    shared-memory lifetime rules). Optional in many pipelines — omit if the
    compiler inserts deallocs automatically.
    """
    handle, _ = _memdesc_handle_and_element_ty(smem, "smem_dealloc")
    _semantic.builder.create_local_dealloc(handle)


__all__ = [
    "SMEM_GENERIC_POINTER_ADDR_SPACE",
    "SharedMemDescType",
    "SharedMemoryDesc",
    "allocate_smem",
    "smem_index",
    "smem_load",
    "smem_store",
    "smem_get_ptr",
    "get_smem_shared_address_u32",
    "smem_dealloc",
]
