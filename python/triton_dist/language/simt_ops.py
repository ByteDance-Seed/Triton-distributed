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
from typing import Sequence
import triton
from triton.language import core as tl
from triton.language.core import builtin, base_type, constexpr, dtype, get_int_dtype
from triton._C.libtriton import ir
import builtins
from triton.language import core as tlc
from typing import List, Optional
from triton_dist._plugin import require_ext_module as _ext


class vector_type(base_type):

    def __init__(self, type, vec_size):
        self.vec_size = vec_size
        assert vec_size > 0 and vec_size == triton.next_power_of_2(vec_size), "vec_size must be power of 2"
        self.elem_type = type
        self.name = f"{self.elem_type}_v{self.vec_size}"
        self.element_nbits = self.elem_type.primitive_bitwidth
        self.vector_nbits = self.element_nbits * self.vec_size

    def __str__(self):
        return self.name

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]):
        for i in range(self.vec_size):
            self.elem_type._flatten_ir_types(builder, out)

    def __eq__(self, other):
        return type(self) is type(other) and self.elem_type == other.elem_type and self.vec_size == other.vec_size

    def mangle(self):
        return 'VECTOR_' + self.name

    def _unflatten_ir(self, handles: List[ir.value], cursor: int):
        values = []
        for i in range(self.vec_size):
            value, cursor = self.elem_type._unflatten_ir(handles, cursor)
            values.append(value)
        return vector(values), cursor


@builtin
def vector_binOp(x, y, op, _semantic=None):
    assert isinstance(x, vector), f"expected vector, got {type(x)}"

    if isinstance(y, vector):
        ret = []
        for a, b in zip(x.values, y.values):
            ret.append(op(a, b, _semantic=_semantic))
        return vector(ret)
    elif isinstance(y, (tl.tensor, tl.constexpr)):
        return vector([op(val, y, _semantic=_semantic) for val in x])
    else:
        raise ValueError(f"expected vector or tensor, got {type(y)}")


@builtin
def vector_add(x, y, _semantic=None):
    return vector_binOp(x, y, tl.add, _semantic=_semantic)


@builtin
def vector_sub(x, y, _semantic=None):
    return vector_binOp(x, y, tl.sub, _semantic=_semantic)


@builtin
def vector_mul(x, y, _semantic=None):
    return vector_binOp(x, y, tl.mul, _semantic=_semantic)


# why inherit from tl.tensor instead of base_value?
# because in code generator, only tl.tensor support binOp(e.g. __add__)
# don't want to introduce more modifications(patch) to code generator
class vector(tl.tensor):
    __triton_builtin__ = True

    def __init__(self, args: Sequence):
        self.values = [i for i in args]

        for val in self.values:
            assert isinstance(val, tl.tensor), f"val = {val}, type = {type(val)}"
            assert val.dtype == self.values[0].dtype

        self.type = vector_type(self.values[0].dtype, len(self.values))

    def __getitem__(self, idx: constexpr):
        if isinstance(idx, int):
            idx = constexpr(idx)
        if isinstance(idx, constexpr):
            return self.values[idx]
        else:
            assert isinstance(idx, (slice, builtins.slice))
            return vector(self.values[idx.start:idx.stop:idx.step])

    def __setitem__(self, idx: constexpr, value):
        if isinstance(idx, int):
            idx = constexpr(idx)
        assert isinstance(idx, constexpr)
        self.values[idx] = value

    @builtin
    def __add__(self, other, _semantic=None):
        return vector_add(self, other, _semantic=_semantic)

    @builtin
    def __sub__(self, other, _semantic=None):
        return vector_sub(self, other, _semantic=_semantic)

    @builtin
    def __mul__(self, other, _semantic=None):
        return vector_mul(self, other, _semantic=_semantic)

    def __eq__(self, other):
        if not isinstance(other, vector):
            return False

        if len(other.values) == len(self.values):
            for a, b in zip(self.values, other.values):
                if a != b:
                    return False
            return True
        return False

    def __hash__(self):
        return hash(builtins.tuple(self.values))

    def __str__(self):
        return "vector " + str([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def _flatten_ir(self, handles: List[ir.value]):
        for v in self.values:
            v._flatten_ir(handles)

    def _set_name(self, builder: ir.builder, name: str):
        # vector is an aggregate of scalar tensors and has no single ``handle``.
        # Triton 3.7.1's code generator names loop-/if-carried values via
        # ``_set_name`` (base ``tl.tensor`` implements it on ``self.handle``), so
        # forward naming to each underlying scalar handle instead.
        for v in self.values:
            v._set_name(builder, name)

    def __repr__(self):
        return f"({' ,'.join(repr(x) for x in self.values)})"

    # bitcast
    @builtin
    def recast(self, new_elem_dtype: dtype, _semantic=None):
        old_elem_dtype = self.values[0].dtype
        old_nbits = old_elem_dtype.primitive_bitwidth
        new_nbits = new_elem_dtype.primitive_bitwidth

        if old_nbits == new_nbits:
            return vector([tl.cast(v, new_elem_dtype, bitcast=True, _semantic=_semantic) for v in self.values])

        # TODO(zhengxuegui.0): use more efficient hardware-specific instructions to impl recast
        if old_nbits % new_nbits == 0:
            from triton_dist.language.extra.language_extra import unpack
            ratio = old_nbits // new_nbits
            new_values = []
            for v in self.values:
                int_ty = get_int_dtype(old_nbits, False)
                int_old = tl.cast(v, int_ty, _semantic=_semantic)
                # for i in range(ratio):
                #     mask: tl.constexpr = tl.cast((1 << new_nbits) - 1, int_ty, _semantic=_semantic)
                #     shift: tl.constexpr = tl.cast(i * new_nbits, int_ty, _semantic=_semantic)
                #     shifted = int_old.__rshift__(shift, _semantic=_semantic)
                #     piece = shifted.__and__(mask, _semantic=_semantic)
                #     new_val = tl.cast(piece, get_int_dtype(new_nbits, False), _semantic=_semantic)
                #     new_val = tl.cast(new_val, new_elem_dtype, bitcast=True, _semantic=_semantic)
                #     new_values.append(new_val)
                unpack_vals = unpack(int_old, new_elem_dtype, _semantic=_semantic)
                new_values.extend(unpack_vals)
            return vector(new_values)

        elif new_nbits % old_nbits == 0:
            from triton_dist.language.extra.language_extra import pack
            ratio = new_nbits // old_nbits
            if len(self.values) % ratio != 0:
                raise ValueError(f"cannot recast: vec_size={len(self.values)} not divisible by ratio={ratio}")
            new_int_ty = get_int_dtype(new_nbits, False)

            new_values = []
            for i in range(0, len(self.values), ratio):
                # combined = tl.constexpr(0)
                # combined = tl.cast(combined, new_int_ty, bitcast=True, _semantic=_semantic)
                # for j in range(ratio):
                #     old_bits = tl.cast(self.values[i + j], old_int_ty, bitcast=True, _semantic=_semantic)
                #     old_bits = tl.cast(old_bits, new_int_ty, _semantic=_semantic)
                #     shifted = old_bits.__lshift__(j * old_nbits, _semantic=_semantic)
                #     combined = combined.__or__(shifted, _semantic=_semantic)
                combined = vector(self.values[i:i + ratio])
                combined = pack(combined, new_int_ty, _semantic=_semantic)
                new_val = tl.cast(combined, new_elem_dtype, bitcast=True, _semantic=_semantic)
                new_values.append(new_val)
            return vector(new_values)
        else:
            raise ValueError(f"cannot recast from {old_elem_dtype} to {new_elem_dtype}: bitwidth not compatible")

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
        return vector([
            tl.cast(v, dtype, fp_downcast_rounding=fp_downcast_rounding, bitcast=bitcast, _semantic=_semantic)
            for v in self.values
        ])


@builtin
def make_vector(args: Sequence, _semantic=None):
    return vector(args)


def fill_vector(vec_size, dtype, val, _semantic=None):
    assert isinstance(vec_size, tl.constexpr), "vec_size must be a constant"
    val = tl.cast(val, dtype, bitcast=False, _semantic=_semantic)
    return make_vector([val for i in range(vec_size)], _semantic=_semantic)


@builtin
def zeros_vector(vec_size, dtype, _semantic=None):
    assert isinstance(vec_size, tl.constexpr), "vec_size must be a constant"
    val = tl.constexpr(0)
    return fill_vector(vec_size, dtype, val, _semantic=_semantic)


class simt_exec_region:
    """Marker context manager for a SIMT (per-thread) execution region.

    Inside an ``@triton_dist.jit`` kernel, ``with simt_exec_region() as
    (thread_idx, threads_per_block):`` is recognized by triton_dist's
    ``visit_With`` dispatch (installed at import, see ``_install_simt_with_dispatch``).
    The dispatch emits a ``simt.simt_exec_region`` op through the plugin and binds
    the per-thread index / block size. The class itself is never instantiated on
    that path; the trivial enter/exit only keeps it usable as a plain (no-op)
    context manager outside of kernel compilation.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return (None, None)

    def __exit__(self, exc_type, exc_value, traceback):
        return False


# Extension of dist triton: create op to load scalar from tile
def extract(input: tlc.tensor, indices: List, _semantic) -> tlc.tensor:
    dst_indices = []
    for idx in indices:
        if isinstance(idx, tlc.tensor):
            dst_indices.append(idx.handle)
        elif isinstance(idx, tlc.constexpr):
            dst_indices.append(_semantic._convert_elem_to_ir_value(idx, require_i64=False))
        else:
            raise ValueError(f"unsupported tensor index: {idx}")
    ret = _ext().distributed_extract(_semantic.builder, [input.handle, *dst_indices])
    return tlc.tensor(ret, input.dtype)


# Extension of dist triton: create op to store scalar to tile
def insert(input: tlc.tensor, scalar, indices, _semantic) -> tlc.tensor:
    if isinstance(indices, (tlc.tensor, tlc.constexpr)):
        indices = [indices]
    dst_indices = []
    for idx in indices:
        if isinstance(idx, tlc.tensor):
            dst_indices.append(idx.handle)
        elif isinstance(idx, tlc.constexpr):
            dst_indices.append(_semantic._convert_elem_to_ir_value(idx, require_i64=False))
        else:
            raise ValueError(f"unsupported tensor index: {idx}")
    return tlc.tensor(_ext().distributed_insert(_semantic.builder, [scalar.handle, input.handle, *dst_indices]),
                      input.type)


# ---------------------------------------------------------------------------
# SIMT execution region: `with` dispatch
# ---------------------------------------------------------------------------
# Upstream Triton lowers `with` blocks in CodeGenerator.visit_With by simply
# calling the context manager's enter/exit; it neither understands region ops
# nor unpacks tuple `as (a, b)` targets. TLX handles this with a dispatch table
# consulted by an (upstream-fork) WITH_DISPATCH hook. Our pinned Triton 3.7.1 has
# no such hook, so we install the same idea by wrapping visit_With from this
# package at import time: a contained, reversible, Python-level extension (the
# same philosophy as the jit/pipeline hooks) -- no Triton source edit.
#
# For `with simt_exec_region() as (thread_idx, threads_per_block):` we:
#   1. materialize thread_idx / threads_per_block (companion get_thread_id /
#      get_block_size) in the parent block and bind the tuple target,
#   2. create the simt.simt_exec_region op (companion create_simt_exec_region_op,
#      which returns the op + entry block handles -- not expressible via OpInfo),
#   3. emit the body inside the region's entry block,
#   4. terminate the body with block_yield (companion create_block_yield_op) and
#      rebind carried names to the region op's results.
import ast as _ast  # noqa: E402
from triton.compiler import code_generator as _code_generator  # noqa: E402
from triton.compiler.code_generator import (  # noqa: E402
    enter_sub_region as _enter_sub_region, flatten_values_to_ir as _flatten_values_to_ir, unflatten_ir_values as
    _unflatten_ir_values, _is_triton_value)
from triton.language.core import _unwrap_if_constexpr as _unwrap  # noqa: E402

_ORIG_VISIT_WITH = None


def _bind_simt_target(self, node, thread_idx, threads_per_block):
    """Bind the `as (thread_idx, threads_per_block)` target(s) of the with-item."""
    optvars = node.items[0].optional_vars
    if optvars is None:
        return
    if isinstance(optvars, (_ast.Tuple, _ast.List)):
        if len(optvars.elts) != 2:
            raise ValueError("simt_exec_region yields exactly "
                             "(thread_idx, threads_per_block)")
        self.set_value(optvars.elts[0].id, thread_idx)
        self.set_value(optvars.elts[1].id, threads_per_block)
    else:
        self.set_value(optvars.id, tl.tuple([thread_idx, threads_per_block]))


def _bind_simt_exec_region(self, node):
    """Lower `with simt_exec_region() as (tid, bs):` to a simt.simt_exec_region.

    The region carries its own block; values assigned to outer-scope (livein)
    names inside it are threaded through as init args / block args / results so
    they remain valid SSA after the region (mirrors how visit_For/visit_If thread
    loop-carried values). Without this, an assignment like ``acc[i, j] = ...``
    inside the region defines a value in the child region that the enclosing
    ``for`` loop would try to use as a loop-carried operand -- a dominance error.

    The region op + block_yield carry the SIMT dialect TypeID, so they are built
    through the companion module (libtriton_dist_ext), which forwards to the
    plugin; OpInfo callbacks cannot return the op/block handles this needs.
    """
    from triton_dist._plugin import load_ext_module
    ext = load_ext_module()
    if ext is None:
        raise RuntimeError("simt_exec_region requires the Triton-distributed companion module "
                           "(libtriton_dist_ext); build it via `pip install ./python`.")

    builder = self.builder
    # thread id / block size are materialized in the *parent* block (the region
    # is not IsolatedFromAbove, so the body can reference them directly).
    thread_idx = tlc.tensor(ext.distributed_get_thread_id(builder, []), tl.int32)
    threads_per_block = tlc.tensor(ext.distributed_get_block_size(builder, []), tl.int32)
    _bind_simt_target(self, node, thread_idx, threads_per_block)

    with _enter_sub_region(self) as sr:
        liveins, _insert_block = sr
        ip, last_loc = self._get_insertion_point_and_loc()

        # Dry run in a throwaway block to discover which livein names are
        # (re)assigned inside the region: those are the captured/carried values.
        self._set_insertion_point_and_loc(ip, last_loc)
        dummy = builder.create_block()
        builder.set_insertion_point_to_start(dummy)
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        dummy.erase()

        names = []
        init_args = []
        for name in self.local_defs:
            if name in liveins:
                names.append(name)
                init_args.append(liveins[name])

        # Build the region op with the carried values as init args; its entry
        # block gets one argument per carried value.
        self._set_insertion_point_and_loc(ip, last_loc)
        init_tys = [v.type for v in init_args]
        init_handles = _flatten_values_to_ir(init_args)
        simt_op = ext.create_simt_exec_region_op(builder, init_handles)
        block = ext.get_simt_region_entry_block(simt_op)
        block_handles = [block.arg(i) for i in range(len(init_handles))]
        block_args = list(_unflatten_ir_values(block_handles, init_tys))

        # Re-run the body with carried names bound to the block args.
        self.lscope = liveins.copy()
        self.local_defs = {}
        for name, val in zip(names, block_args):
            self.set_value(name, val)
        builder.set_insertion_point_to_start(block)
        builder.create_barrier()
        self.scf_stack.append(node)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()

        # Yield the (possibly updated) carried values in the same order.
        yields = [self.lscope[name] for name in names]
        yield_handles = _flatten_values_to_ir(yields)
        builder.create_barrier()
        ext.create_block_yield_op(builder, yield_handles)

    # Outside the region: rebind carried names to the region op's results.
    result_handles = [simt_op.get_result(i) for i in range(len(init_handles))]
    result_values = list(_unflatten_ir_values(result_handles, init_tys))
    for name, val in zip(names, result_values):
        self.set_value(name, val)


def _patched_visit_With(self, node):
    if len(node.items) == 1:
        ctx = node.items[0].context_expr
        if isinstance(ctx, _ast.Call):
            try:
                fn = _unwrap(self.visit(ctx.func))
            except Exception:
                fn = None
            if fn is simt_exec_region:
                return _bind_simt_exec_region(self, node)
    return _ORIG_VISIT_WITH(self, node)


def _install_simt_with_dispatch():
    """Idempotently wrap CodeGenerator.visit_With to dispatch simt_exec_region."""
    global _ORIG_VISIT_WITH
    current = _code_generator.CodeGenerator.visit_With
    if getattr(current, "_triton_dist_wrapped", False):
        return
    _ORIG_VISIT_WITH = current
    _patched_visit_With._triton_dist_wrapped = True
    _code_generator.CodeGenerator.visit_With = _patched_visit_With


# ---------------------------------------------------------------------------
# SIMT element access: tile[idx...] load / store
# ---------------------------------------------------------------------------
# Inside a simt_exec_region, `tile[i, j]` reads and `tile[i, j] = v` writes a
# single (per-thread) scalar element of a block tensor -- lowered to the plugin's
# distributed_extract / distributed_insert ops. Upstream Triton's
# tensor.__getitem__ rejects integer indices and visit_Subscript_Store raises
# NotImplementedError, so we route these through extract()/insert() by wrapping
# the two CodeGenerator subscript hooks (visit_Subscript / assignTarget dispatch
# to them dynamically, so patching these two methods covers both paths). This is
# the Python-level counterpart to the fork's intrusive code_generator edits.
_ORIG_VISIT_SUBSCRIPT_LOAD = None
_ORIG_VISIT_SUBSCRIPT_STORE = None


def _as_index_list(slices):
    # A single (non-tuple) index. Note Triton lowers ``x[:]`` to its *own*
    # ``triton.language.core.slice`` (not a builtins.slice), so accept both;
    # otherwise it falls through to ``return slices`` and _is_scalar_index tries
    # to iterate the (non-iterable) slice object -> "'slice' object is not
    # iterable" (e.g. ``offs[:]`` in the allgather/ep_a2a kernels).
    if isinstance(slices, (builtins.slice, tlc.slice, constexpr, tlc.tensor)) or slices is None:
        return [slices]
    if isinstance(slices, tlc.tuple):
        return list(slices.values)
    return slices


def _is_scalar_index(sls):
    """True iff every index is a concrete int (constexpr) or runtime scalar
    (tensor), i.e. per-thread element access rather than block slicing."""
    if not sls:
        return False
    for s in sls:
        if not isinstance(s, (constexpr, tlc.tensor)):
            return False
        if isinstance(s, constexpr) and s.value is None:
            return False
    return True


def _patched_visit_Subscript_Load(self, node):
    lhs = self.visit(node.value)
    slices = self.visit(node.slice)
    # Only divert *per-thread scalar* element reads (``tile[i, j]`` with int /
    # runtime-scalar indices) to distributed_extract. Everything else -- block
    # slicing (``offs[:]``, ``x[None, :]``), aggregates, plain Python objects --
    # must follow upstream's path unchanged (note: upstream goes through
    # call_Method, which is what makes a bare ``slice`` work; calling
    # ``__getitem__`` directly does not).
    if isinstance(lhs, tlc.tensor):
        sls = _as_index_list(slices)
        if _is_scalar_index(sls):
            return extract(lhs, sls, self.semantic)
    if _is_triton_value(lhs):
        return self.call_Method(node, lhs.__getitem__, lhs, [slices], {})
    return lhs[slices]


def _patched_visit_Subscript_Store(self, node, value):
    lhs = self.visit(node.value)
    # Only divert *per-thread scalar* element writes (``tile[i, j] = v`` with int /
    # runtime-scalar indices) to distributed_insert -- symmetric with the load path.
    # Block-slice stores fall through to upstream (which raises NotImplementedError).
    if isinstance(lhs, tlc.tensor):
        sls = _as_index_list(self.visit(node.slice))
        if _is_scalar_index(sls):
            # ``insert`` returns a *new* SSA value that must rebind the target name;
            # only a bare ``Name`` target (``t[i] = v``) can be rebound this way.
            if not isinstance(node.value, _ast.Name):
                raise ValueError("SIMT element assignment `t[i] = v` requires the target tensor to be a "
                                 f"simple variable name; got {type(node.value).__name__}.")
            ret = insert(lhs, value, sls, self.semantic)
            self.set_value(node.value.id, ret)
            return
    return _ORIG_VISIT_SUBSCRIPT_STORE(self, node, value)


def _install_simt_subscript_dispatch():
    """Idempotently route tensor subscript load/store through extract/insert."""
    global _ORIG_VISIT_SUBSCRIPT_LOAD, _ORIG_VISIT_SUBSCRIPT_STORE
    load = _code_generator.CodeGenerator.visit_Subscript_Load
    store = _code_generator.CodeGenerator.visit_Subscript_Store
    if getattr(load, "_triton_dist_wrapped", False):
        return
    _ORIG_VISIT_SUBSCRIPT_LOAD = load
    _ORIG_VISIT_SUBSCRIPT_STORE = store
    _patched_visit_Subscript_Load._triton_dist_wrapped = True
    _code_generator.CodeGenerator.visit_Subscript_Load = _patched_visit_Subscript_Load
    _code_generator.CodeGenerator.visit_Subscript_Store = _patched_visit_Subscript_Store


_install_simt_with_dispatch()
_install_simt_subscript_dispatch()
