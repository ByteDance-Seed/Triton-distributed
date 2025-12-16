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

    def __init__(self, _builder=None):
        self._builder = _builder

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


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
    ret = _semantic.builder.create_extract(input.handle, dst_indices)
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
    return tlc.tensor(_semantic.builder.create_insert(scalar.handle, input.handle, dst_indices), input.type)
