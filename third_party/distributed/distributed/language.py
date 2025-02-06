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
from triton.language import core as tlc
from triton.language.semantic import _str_to_sem, _str_to_scope, to_tensor, _convert_elem_to_ir_value
from triton.language.core import builtin
from triton._C.libtriton import ir


def _str_to_dist_signal_op(sig_op):
    cache = None
    if sig_op:
        if sig_op == "set":
            cache = ir.SIGNAL_OP.SET
        elif sig_op == "add":
            cache = ir.SIGNAL_OP.ADD
        else:
            raise ValueError(f"Signal Op {sig_op} not supported")
    return cache


def _str_to_dist_comm_scopre(comm_scope):
    scope = None
    if comm_scope:
        if comm_scope == "gpu":
            scope = ir.COMM_SCOPE.GPU
        elif comm_scope == "intra_node":
            scope = ir.COMM_SCOPE.INTRA_NODE
        elif comm_scope == "inter_node":
            scope = ir.COMM_SCOPE.INTER_NODE
        else:
            raise ValueError(f"Comm Scope {comm_scope} not supported")
    return scope


@builtin
def wait(barrierPtrs, numBarriers, scope: str, semantic: str, waitValue: int = 1, _builder=None):
    if not barrierPtrs.type.scalar.is_ptr():
        raise ValueError(f"Unsupported barrierPtrs type {barrierPtrs.type.__repr__()} in `distributed.language.wait`")
    elem_ty = barrierPtrs.dtype.element_ty
    require_i64 = False
    if elem_ty.is_int64() or elem_ty.is_uint64():
        require_i64 = True
    waitValue = _convert_elem_to_ir_value(_builder, waitValue, require_i64=require_i64)
    scope = _str_to_scope(scope)
    semantic = _str_to_sem(semantic)
    return tlc.tensor(
        _builder.create_distributed_wait(barrierPtrs.handle,
                                         to_tensor(numBarriers, _builder).handle, waitValue, scope, semantic,
                                         tlc.int32.to_ir(_builder)), tlc.int32)


@builtin
def consume_token(value, token, _builder=None):
    assert token.type.scalar.is_int(), "token must be of int type"
    handle = _builder.create_distributed_consume_token(value.handle, token.handle)
    if isinstance(value, tlc.tensor_descriptor):
        return tlc.tensor_descriptor(handle, value.shape, value.strides, value.block_type)
    else:
        return tlc.tensor(handle, value.type)


@builtin
def rank(axis=-1, _builder=None):
    axis = _convert_elem_to_ir_value(_builder, axis, require_i64=False)
    return tlc.tensor(_builder.create_get_rank(axis), tlc.int32)


@builtin
def num_ranks(axis=-1, _builder=None):
    axis = _convert_elem_to_ir_value(_builder, axis, require_i64=False)
    return tlc.tensor(_builder.create_get_num_ranks(axis), tlc.int32)


@builtin
def symm_at(ptr, rank, _builder=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    rank = _convert_elem_to_ir_value(_builder, rank, require_i64=False)
    return tlc.tensor(_builder.create_symm_at(ptr.handle, rank), ptr.type)


@builtin
def notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    assert ptr.dtype.element_ty == tlc.uint64, "the dtype of signal ptr should be uint64"

    rank = _convert_elem_to_ir_value(_builder, rank, require_i64=False)
    signal = _convert_elem_to_ir_value(_builder, signal, require_i64=True)
    sig_op = _str_to_dist_signal_op(sig_op)
    comm_scope = _str_to_dist_comm_scopre(comm_scope)
    return tlc.tensor(_builder.create_notify(ptr.handle, signal, rank, sig_op, comm_scope), tlc.void)
