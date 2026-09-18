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
from triton.language.core import builtin
from triton_dist._plugin import require_ext_module as _ext

# Distributed ops are provided by the out-of-tree Triton plugin
# (libtriton_dist.so, dialects + passes) and exposed to Python through the
# companion module as `ext.distributed_*(builder, [operands...])`. They are NOT
# OpInfo builder methods: on the pinned Triton 3.7.1 a plugin-op binding inserts
# no result slot and returns no Value (see plugin/lib/ext/py_module.cpp). Enum
# attributes are passed as i32 constant Values whose integer encodings match the
# dialect TableGen enums.

# MemSyncScope (triton core): GPU=1, CTA=2, SYSTEM=3
_SCOPE_TO_INT = {"gpu": 1, "cta": 2, "sys": 3}
# MemSemantic (triton core): RELAXED=1, ACQUIRE=2, RELEASE=3, ACQUIRE_RELEASE=4
_SEM_TO_INT = {"relaxed": 1, "acquire": 2, "release": 3, "acq_rel": 4}
# distributed::SignalOp: SET=1, ADD=2
_SIGNAL_OP_TO_INT = {"set": 1, "add": 2}
# distributed::CommScope: GPU=1, INTRA_NODE=2, INTER_NODE=3
_COMM_SCOPE_TO_INT = {"gpu": 1, "intra_node": 2, "inter_node": 3}


def _unwrap_str(v):
    """Accept a raw str or a triton ``constexpr`` wrapping a str."""
    if isinstance(v, tlc.constexpr):
        v = v.value
    return v


def _str_to_scope_int(scope_option):
    scope_option = _unwrap_str(scope_option)
    if not scope_option:
        return _SCOPE_TO_INT["gpu"]
    if scope_option not in _SCOPE_TO_INT:
        raise ValueError(f"Memory scope {scope_option} not supported")
    return _SCOPE_TO_INT[scope_option]


def _str_to_sem_int(sem_option):
    sem_option = _unwrap_str(sem_option)
    if not sem_option:
        return _SEM_TO_INT["acq_rel"]
    if sem_option not in _SEM_TO_INT:
        raise ValueError(f"Memory semantic {sem_option} not supported")
    return _SEM_TO_INT[sem_option]


def _str_to_dist_signal_op_int(sig_op):
    sig_op = _unwrap_str(sig_op)
    if sig_op not in _SIGNAL_OP_TO_INT:
        raise ValueError(f"Signal Op {sig_op} not supported")
    return _SIGNAL_OP_TO_INT[sig_op]


def _str_to_dist_comm_scope_int(comm_scope):
    comm_scope = _unwrap_str(comm_scope)
    if comm_scope not in _COMM_SCOPE_TO_INT:
        raise ValueError(f"Comm Scope {comm_scope} not supported")
    return _COMM_SCOPE_TO_INT[comm_scope]


@builtin
def wait(barrierPtrs, numBarriers, scope: str, semantic: str, waitValue: int = 1, _semantic=None):
    if not barrierPtrs.type.scalar.is_ptr():
        raise ValueError(f"Unsupported barrierPtrs type {barrierPtrs.type.__repr__()} in `distributed.language.wait`")
    elem_ty = barrierPtrs.dtype.element_ty
    require_i64 = False
    if elem_ty.is_int64() or elem_ty.is_uint64():
        require_i64 = True
    waitValue = _semantic._convert_elem_to_ir_value(waitValue, require_i64=require_i64)
    scope_const = _semantic.builder.get_int32(_str_to_scope_int(scope))
    sem_const = _semantic.builder.get_int32(_str_to_sem_int(semantic))
    handle = _ext().distributed_wait(
        _semantic.builder,
        [barrierPtrs.handle,
         _semantic.to_tensor(numBarriers).handle, waitValue, scope_const, sem_const])
    return tlc.tensor(handle, tlc.int32)


@builtin
def consume_token(value, token, _semantic=None):
    assert token.type.scalar.is_int(), "token must be of int type"
    # A `_block_ptr` (from tl.make_block_ptr) is a pure-frontend object in Triton
    # 3.7.1 -- it has no single MLIR handle, only a scalar `base` pointer tensor plus
    # shape/strides/offsets materialised at load time. Thread the token dependency
    # through that base and return an otherwise-identical block pointer so the
    # subsequent tl.load observes the wait via a data dependency.
    if getattr(value, "__triton_block_ptr__", False):
        base_handle = _ext().distributed_consume_token(_semantic.builder, [value.base.handle, token.handle])
        new_base = tlc.tensor(base_handle, value.base.type)
        # Rebuild via the public constructor (mirrors make_block_ptr); the already
        # canonicalized shape/strides/offsets pass through idempotently.
        return tlc._block_ptr(new_base, value.shape, value.strides, value.offsets, value.block_shape, value.order,
                              _semantic=_semantic)
    handle = _ext().distributed_consume_token(_semantic.builder, [value.handle, token.handle])
    if isinstance(value, tlc.tensor_descriptor):
        return tlc.tensor_descriptor(handle, value.shape, value.strides, value.block_type)
    else:
        return tlc.tensor(handle, value.type)


@builtin
def rank(axis=-1, _semantic=None):
    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tlc.tensor(_ext().distributed_get_rank(_semantic.builder, [axis]), tlc.int32)


@builtin
def num_ranks(axis=-1, _semantic=None):
    axis = _semantic._convert_elem_to_ir_value(axis, require_i64=False)
    return tlc.tensor(_ext().distributed_get_num_ranks(_semantic.builder, [axis]), tlc.int32)


@builtin
def symm_at(ptr, rank, _semantic=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    rank = _semantic._convert_elem_to_ir_value(rank, require_i64=False)
    return tlc.tensor(_ext().distributed_symm_at(_semantic.builder, [ptr.handle, rank]), ptr.type)


@builtin
def notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _semantic=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    assert ptr.dtype.element_ty == tlc.uint64 or ptr.dtype.element_ty == tlc.int64, "the dtype of signal ptr should be uint64"

    rank = _semantic._convert_elem_to_ir_value(rank, require_i64=False)
    signal = _semantic._convert_elem_to_ir_value(signal, require_i64=True)
    sig_op_const = _semantic.builder.get_int32(_str_to_dist_signal_op_int(sig_op))
    comm_scope_const = _semantic.builder.get_int32(_str_to_dist_comm_scope_int(comm_scope))
    _ext().distributed_notify(_semantic.builder, [ptr.handle, signal, rank, sig_op_const, comm_scope_const])
    return tlc.tensor(None, tlc.void)
