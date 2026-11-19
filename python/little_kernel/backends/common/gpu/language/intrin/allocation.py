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
"""
Memory allocation intrinsics (local/shared/dynamic, zeros/full).

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4, str_
from little_kernel.language.intrin import identifiers as contracts

_strip_quotes = contracts.strip_codegen_identifier


def codegen_alloc_dynamic_shared_memory(size_bytes, align_bytes, name, dyn_exprs=None):
    # Strip C++ string literal quotes if present (name is a variable identifier)
    var_name = name.strip('"')
    stmt = f"extern __shared__ __align__({align_bytes}) uint8_t {var_name}[]"

    return Builtin(body="", includes=[], return_val=stmt)


def alloc_dynamic_shared_memory_eval_arg_type(ctx, size_bytes, align_bytes=None, name=None, dyn_exprs=None):
    """Type-check a dynamic shared memory allocation call site.

    This builtin is inserted by the ``insert_mem_alloc`` pass with the
    internal signature ``(size_bytes, align_bytes, name, dyn_exprs)``; it is
    not a general-purpose user allocation idiom.  Calling it directly with a
    dtype-first pair such as ``ll.alloc_dynamic_shared_memory(ll.uint8,
    4096)`` binds the dtype to ``size_bytes`` and leaves ``name`` unset,
    which previously surfaced as an opaque Python ``TypeError`` (``missing 1
    required positional argument: 'name'``) wrapped in a generic
    "Failed to compute return type" diagnostic.  Validate the call shape
    here and raise a typed diagnostic that names the intended public idiom
    instead.
    """
    if isinstance(size_bytes, LLType) or name is None or not isinstance(align_bytes, int):
        from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
        raise TypeInferenceError("alloc_dynamic_shared_memory is an internal builtin inserted by "
                                 "the insert_mem_alloc pass with signature (size_bytes, "
                                 "align_bytes, name, dyn_exprs); it must not be called directly "
                                 "from a kernel. To allocate dynamic shared memory with an "
                                 "element type, declare it through the typed dynamic-shared "
                                 "idiom instead, e.g. "
                                 "smem = ll.empty([elems], ll.uint8, scope='dynamic_shared') "
                                 "(optionally after ll.align_memory(bytes, 'dynamic_shared')), "
                                 "then take its address with "
                                 "addr: ll.uint32 = ll.cvta_generic_to_shared(smem) when a "
                                 "mbarrier or TMA operand needs a .shared window address.")
    ctx[name] = uint8


@builtin(eval_return_type=void, eval_arg_type=alloc_dynamic_shared_memory_eval_arg_type,
         codegen_func=codegen_alloc_dynamic_shared_memory)
def alloc_dynamic_shared_memory(size_bytes: int, align_bytes: int, name: str, dyn_exprs: str = ""):
    """Allocate dynamic shared memory with the given size in bytes.

    size_bytes is the compile-time (static) byte total; dyn_exprs, when
    present, is a serialized list of runtime-sized (shape-expression)
    allocation descriptors attached by insert_mem_alloc — consumed only by
    the launch wrapper's host-side reservation/legality checks, never by
    kernel codegen.
    """
    raise RuntimeError("alloc_dynamic_shared_memory should never be called in compilation")


def codegen_alloc_local_memory(var_name, dtype, elems):
    var_name = _strip_quotes(var_name)
    stmt = f"{dtype} {var_name}[{elems}]"

    return Builtin(body="", includes=[], return_val=stmt)


def _local_alloc_is_direct_form(first) -> bool:
    """Discriminate the two spellings of the local-alloc family.

    Internal form (inserted by insert_mem_alloc, or spelled with a quoted
    name like the shared-memory builtins): ``alloc_local_memory(
    <var_name:str>, dtype, elems, ...)`` — ``first`` is the buffer name.

    Direct assignment form: ``v = alloc_local_memory(dtype, elems, ...)`` —
    the assignment target supplies the buffer name and ``first`` is the
    element dtype itself.  The two views of ``first`` this helper serves:
      * eval_arg_type receives resolved static args — a Python ``str`` name
        (internal) vs. an ``LLType`` dtype (direct);
      * eval_return_type receives *inferred* argument types — the ``str_``
        LLType of the name literal (internal) vs. the dtype LLType itself
        (direct).
    """
    if not isinstance(first, LLType):
        # Resolved static view: a Python str buffer name -> internal form.
        return False
    # Inferred-type view: the name literal's own type is `str_`; every dtype
    # argument (ll.float32 etc.) infers to the dtype LLType itself, never to
    # `str_`, because no dtype is the string type.
    return first != str_


def alloc_local_memory_eval_arg_type(ctx, first, second, elems=None):
    if _local_alloc_is_direct_form(first):
        # Direct form: (dtype, elems).  There is no name to bind here; the
        # assigned/annotated target becomes the array name at codegen, and
        # statement type inference binds it to the call's return type.
        return
    # Use Tensor[dtype] so that runtime subscript (e.g. arr[stage]) correctly
    # infers element type.  Consistent with alloc_shared_memory_eval_arg_type.
    ctx[first] = Tensor[second]


def _alloc_local_memory_return_type(first, second, elems=None):
    """Return the tensor type for both spellings.

    eval_return_type receives *inferred argument types*: the dtype argument
    is the LLType itself in the direct form (position 1) and the internal
    form (position 2); see _local_alloc_is_direct_form.
    """
    if _local_alloc_is_direct_form(first):
        return Tensor[first]
    return Tensor[second]


@builtin(eval_return_type=_alloc_local_memory_return_type, eval_arg_type=alloc_local_memory_eval_arg_type,
         codegen_func=codegen_alloc_local_memory)
def alloc_local_memory(var_name: str, dtype: LLType, elems: int):
    """Allocate local memory with the given number of elements.

    Also usable directly in an assignment where the target supplies the
    buffer name: ``v: ll.float32 = ll.alloc_local_memory(ll.float32, N)``
    declares an N-element local (register/local-memory) array whose elements
    are addressable as ``v[i]``.
    """
    raise RuntimeError("alloc_local_memory should never be called in compilation")


def codegen_alloc_shared_memory(var_name, dtype, elems, align_bytes="0"):
    """Generate static shared memory declaration: __shared__ dtype var_name[elems]

    If align_bytes > 0, generates: __shared__ __align__(align_bytes) dtype var_name[elems]
    """
    var_name = _strip_quotes(var_name)
    # align_bytes comes in as a string from codegen
    try:
        align_val = int(align_bytes)
    except (ValueError, TypeError):
        align_val = 0

    # var_name arrives as a codegened C string literal (e.g. '"s_rank"');
    # strip the quotes so the declaration is valid C++ (same convention as
    # codegen_alloc_local_memory above).
    if isinstance(var_name, str):
        var_name = var_name.strip('"').strip("'")

    if align_val > 0:
        stmt = f"__shared__ __align__({align_val}) {dtype} {var_name}[{elems}]"
    else:
        stmt = f"__shared__ {dtype} {var_name}[{elems}]"
    return Builtin(body="", includes=[], return_val=stmt)


def alloc_shared_memory_eval_arg_type(ctx, var_name, dtype, elems, align_bytes=0):
    """Set variable type in context for shared memory allocation."""
    ctx[var_name] = Tensor[dtype]


@builtin(eval_return_type=lambda var_name, dtype, elems, align_bytes=0: Tensor[dtype],
         eval_arg_type=alloc_shared_memory_eval_arg_type, codegen_func=codegen_alloc_shared_memory)
def alloc_shared_memory(var_name: str, dtype: LLType, elems: int, align_bytes: int = 0):
    """Allocate static shared memory with the given number of elements.

    Args:
        var_name: Variable name for the shared memory
        dtype: Data type of elements
        elems: Number of elements
        align_bytes: Alignment in bytes (0 means no alignment specified)
    """
    raise RuntimeError("alloc_shared_memory should never be called in compilation")


def codegen_slice_dynamic_shared_memory(var_name, dtype, start, size, shmem_buf_name):
    # TODO(zhengsize): find a better way to handle such hacky case
    # See codegen_alloc_shared_memory: strip C string literal quotes from
    # Constant-codegened names so the declaration names the raw identifier.
    var_name = var_name.strip('"')
    if '[' in var_name and ']' in var_name:
        stmt = f"{var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"
    else:
        stmt = f"{dtype} {var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"

    return Builtin(body="", includes=[], return_val=stmt)


def slice_dynamic_shared_memory_eval_arg_type(ctx, var_name, dtype, start, size, shmem_buf_name):
    ctx[var_name] = dtype


@builtin(eval_return_type=lambda var_name, dtype, start, size, shmem_buf_name: dtype,
         eval_arg_type=slice_dynamic_shared_memory_eval_arg_type, codegen_func=codegen_slice_dynamic_shared_memory)
def slice_dynamic_shared_memory(var_name: str, dtype: LLType, start: int, size: int, shmem_buf_name: str):
    """Slice dynamic shared memory with the given start and size."""
    raise RuntimeError("slice_dynamic_shared_memory should never be called in compilation")


def codegen_align_memory(align_bytes, scope):
    return Builtin(body="", includes=[], return_val=f"/* align {scope} to {align_bytes} */")


@builtin(eval_return_type=void, codegen_func=codegen_align_memory)
def align_memory(align_bytes: int, scope: str):
    """Align memory to the given byte alignment."""
    raise RuntimeError("align_memory should never be called in compilation")


def codegen_empty(shape, dtype, scope):
    raise RuntimeError("empty should never be called in code generation")


@builtin(eval_return_type=lambda shape, dtype, scope, align=None, max_elems=None: Tensor[dtype],
         codegen_func=codegen_empty)
def empty(shape: list, dtype: LLType, scope: str, align: int = None, max_elems: int = None):
    """Create an empty tensor with the given shape, dtype, and scope.

    Optional align pads this buffer's start offset within its scope
    (power-of-two bytes, not for scope local).

    For scope="dynamic_shared" only, a shape entry may alternatively be a
    launch shape-expression string (the same mini-language as launch.grid in
    the export spec); the reservation is then host-evaluated at launch time.
    Such a runtime-sized allocation requires `max_elems`, a static positive
    legality bound on its total element count, and must be the last
    dynamic_shared allocation in the kernel (later offsets would be unknown
    at compile time). Kernel code accesses the buffer through the returned
    tensor with runtime offsets as usual.
    """
    raise RuntimeError("empty should never be called in compilation")


# Zero literal for C++ codegen (decl+init in one)
_ZERO_LITERAL = {
    "float": "0.0f",
    "float32": "0.0f",
    "double": "0.0",
    "float64": "0.0",
    # Integer types use 0 (not 0.0f)
    "int": "0",
    "int32": "0",
    "int32_t": "0",
    "int64": "0",
    "int64_t": "0",
    "uint32": "0",
    "uint32_t": "0",
    "uint64": "0",
    "uint64_t": "0",
    "int8": "0",
    "int8_t": "0",
    "uint8": "0",
    "uint8_t": "0",
    "int16": "0",
    "int16_t": "0",
    "uint16": "0",
    "uint16_t": "0",
}

# Threshold for using initializer list vs loop
# For small arrays, initializer list is more efficient (no loop overhead)
# For large arrays, loop is more readable and compiler can optimize
_INIT_LIST_THRESHOLD = 8


def codegen_alloc_local_zeros(var_name, dtype_cpp, elems):
    """Emit local decl + zero init.

    For small arrays (<=8 elements), use initializer list: float arr[2] = {0.0f, 0.0f};
    For large arrays, use loop: float arr[64]; for(int _i=0; _i<64; _i++) arr[_i]=0.0f;

    Initializer list is more efficient for small arrays as it avoids loop overhead
    and allows compiler to better optimize register allocation.
    """
    var_name = _strip_quotes(var_name)
    dtype_str = str(dtype_cpp)
    # Determine default based on type name (integer types use 0, float types use 0.0f)
    if dtype_str in _ZERO_LITERAL:
        zero = _ZERO_LITERAL[dtype_str]
    elif 'int' in dtype_str.lower() or 'uint' in dtype_str.lower():
        zero = "0"
    else:
        zero = "0.0f"
    try:
        n = int(elems)
    except (ValueError, TypeError):
        n = 100  # Fallback to loop for dynamic sizes

    if n <= _INIT_LIST_THRESHOLD:
        # Use initializer list for small arrays
        init_list = ", ".join([zero] * n)
        return Builtin(
            body="",
            includes=[],
            return_val=f"{dtype_cpp} {var_name}[{elems}] = {{{init_list}}}",
        )
    else:
        # Use loop for large arrays with #pragma unroll for better optimization
        return Builtin(
            body="",
            includes=[],
            return_val=
            f"{dtype_cpp} {var_name}[{elems}];\n#pragma unroll\nfor(int _i=0; _i<{elems}; _i++) {var_name}[_i]={zero}",
        )


def alloc_local_zeros_eval_arg_type(ctx, first, second, elems=None):
    if _local_alloc_is_direct_form(first):
        return
    ctx[first] = Tensor[second]


@builtin(
    eval_return_type=_alloc_local_memory_return_type,
    eval_arg_type=alloc_local_zeros_eval_arg_type,
    codegen_func=codegen_alloc_local_zeros,
)
def alloc_local_zeros(var_name: str, dtype: LLType, elems: int):
    """Allocate local memory and zero it (internal: inserted by pass for x = ll.zeros(..., scope=\"local\")).

    Also usable directly in an assignment where the target supplies the
    buffer name: ``v: ll.float32 = ll.alloc_local_zeros(ll.float32, N)``.
    """
    raise RuntimeError("alloc_local_zeros should never be called in compilation")


def codegen_zeros(shape, dtype, scope):
    """User-facing zeros is replaced by pass for local scope; never codegen directly."""
    raise RuntimeError("ll.zeros must be used as x = ll.zeros(..., scope=\"local\") and is replaced by the pass")


@builtin(eval_return_type=lambda shape, dtype, scope, align=None: Tensor[dtype], codegen_func=codegen_zeros)
def zeros(shape: list, dtype: LLType, scope: str, align: int = None):
    """Allocate and zero-initialize (same shape/dtype/scope as empty). Use x = ll.zeros(...); one intrinsic, no separate init. Optional align as in ll.empty."""
    raise RuntimeError("zeros should never be called in compilation")


# ==============================================================================
# Full array initialization (like numpy.full)
# ==============================================================================


def codegen_alloc_local_full(var_name, dtype_cpp, elems, fill_value):
    """Emit local decl + fill init.

    For small arrays (<=8 elements), use initializer list: float arr[2] = {val, val};
    For large arrays, use loop: float arr[64]; for(int _i=0; _i<64; _i++) arr[_i]=val;

    Special handling for string fill values:
    - "-INFINITY" -> -INFINITY (C++ macro)
    - "INFINITY" -> INFINITY (C++ macro)
    - Other strings are stripped of quotes
    """
    var_name = _strip_quotes(var_name)
    try:
        n = int(elems)
    except (ValueError, TypeError):
        n = 100  # Fallback to loop for dynamic sizes

    # Handle string fill values - strip quotes for C++ output
    fill_str = str(fill_value)
    if fill_str.startswith('"') and fill_str.endswith('"'):
        fill_str = fill_str[1:-1]  # Remove surrounding quotes

    # Add cmath include for INFINITY macros
    includes = []
    if 'INFINITY' in fill_str:
        includes = ["<cmath>"]

    if n <= _INIT_LIST_THRESHOLD:
        # Use initializer list for small arrays
        init_list = ", ".join([fill_str] * n)
        return Builtin(
            body="",
            includes=includes,
            return_val=f"{dtype_cpp} {var_name}[{elems}] = {{{init_list}}}",
        )
    else:
        # Use loop for large arrays with #pragma unroll for better optimization
        return Builtin(
            body="",
            includes=includes,
            return_val=
            f"{dtype_cpp} {var_name}[{elems}];\n#pragma unroll\nfor(int _i=0; _i<{elems}; _i++) {var_name}[_i]={fill_str}",
        )


def alloc_local_full_eval_arg_type(ctx, first, second, elems=None, fill_value=None):
    if _local_alloc_is_direct_form(first):
        return
    ctx[first] = Tensor[second]


def _alloc_local_full_return_type(first, second, elems=None, fill_value=None):
    if _local_alloc_is_direct_form(first):
        return Tensor[first]
    return Tensor[second]


@builtin(
    eval_return_type=_alloc_local_full_return_type,
    eval_arg_type=alloc_local_full_eval_arg_type,
    codegen_func=codegen_alloc_local_full,
)
def alloc_local_full(var_name: str, dtype: LLType, elems: int, fill_value):
    """Allocate local memory and fill with value (internal: inserted by pass for x = ll.full(..., scope=\"local\")).

    Also usable directly in an assignment where the target supplies the
    buffer name: ``v: ll.float32 = ll.alloc_local_full(ll.float32, N, f)``.
    """
    raise RuntimeError("alloc_local_full should never be called in compilation")


def codegen_full(shape, dtype, scope, fill_value):
    """User-facing full is replaced by pass for local scope; never codegen directly."""
    raise RuntimeError("ll.full must be used as x = ll.full(..., scope=\"local\") and is replaced by the pass")


@builtin(eval_return_type=lambda shape, dtype, scope, fill_value, align=None: Tensor[dtype], codegen_func=codegen_full)
def full(shape: list, dtype: LLType, scope: str, fill_value, align: int = None):
    """Allocate and fill-initialize with given value. Use x = ll.full([n], dtype, scope, value). Optional align as in ll.empty (fill_value stays positional fourth)."""
    raise RuntimeError("full should never be called in compilation")


from little_kernel.language.builtin_base import register_by_value_arguments, register_element_annotation_alloc

register_by_value_arguments(alloc_shared_memory, ("elems", "align_bytes"))
register_by_value_arguments(alloc_local_memory, ("elems", ))
register_by_value_arguments(alloc_dynamic_shared_memory, ("size_bytes", "align_bytes"))
# Local declaration allocs additionally accept the element-annotation
# spelling: v: ll.float32 = ll.alloc_local_*(ll.float32, N[, ...]) binds an
# N-element array under the assignment target's own name.
register_element_annotation_alloc(alloc_local_memory)
register_element_annotation_alloc(alloc_local_zeros)
register_element_annotation_alloc(alloc_local_full)
