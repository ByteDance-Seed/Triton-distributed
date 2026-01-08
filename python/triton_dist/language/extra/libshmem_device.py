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
from .utils import ModuleProxy
from triton_dist.utils import is_cuda, is_rocshmem, is_mori_shmem
import triton_dist.language.extra.cuda.libnvshmem_device as libnvshmem_device
import triton_dist.language.extra.hip.librocshmem_device as librocshmem_device
import triton_dist.language.extra.hip.libmori_shmem_device as libmori_shmem_device

import sys

_shmem_module = ModuleProxy([
    (is_cuda, libnvshmem_device),
    (is_rocshmem, librocshmem_device),
    (is_mori_shmem, libmori_shmem_device),
])


@_shmem_module.dispatch
def set_rocshmem_ctx(ctx):
    """ROCSHMEM only"""
    ...


@_shmem_module.dispatch
def my_pe():
    """Both NVSHMEM and ROCSHMEM"""
    ...


@_shmem_module.dispatch
def n_pes():
    """Both NVSHMEM and ROCSHMEM"""
    ...


@_shmem_module.dispatch
def team_my_pe(team):
    ...


@_shmem_module.dispatch
def team_n_pes(team):
    ...


@_shmem_module.dispatch
def int_p(dest, value, pe, qp_id=0):
    """Both NVSHMEM and ROCSHMEM"""
    ...


@_shmem_module.dispatch
def remote_ptr(local_ptr, pe):
    """Both NVSHMEM and ROCSHMEM"""
    ...


@_shmem_module.dispatch
def remote_mc_ptr(team, ptr):
    ...


@_shmem_module.dispatch
def barrier_all():
    ...


@_shmem_module.dispatch
def barrier_all_block():
    ...


@_shmem_module.dispatch
def barrier_all_warp():
    ...


@_shmem_module.dispatch
def barrier_all_wave():
    ...


@_shmem_module.dispatch
def barrier_all_wg():
    ...


@_shmem_module.dispatch
def barrier(team):
    ...


@_shmem_module.dispatch
def barrier_block(team):
    ...


@_shmem_module.dispatch
def barrier_warp(team):
    ...


@_shmem_module.dispatch
def sync_all():
    ...


@_shmem_module.dispatch
def sync_all_block():
    ...


@_shmem_module.dispatch
def sync_all_warp():
    ...


@_shmem_module.dispatch
def team_sync_block(team):
    ...


@_shmem_module.dispatch
def team_sync_warp(team):
    ...


@_shmem_module.dispatch
def quiet():
    ...

@_shmem_module.dispatch
def quiet_pe():
    ...

@_shmem_module.dispatch
def fence():
    ...


@_shmem_module.dispatch
def getmem_nbi_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_nbi_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_nbi(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_nbi_wave(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_nbi_wg(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_wave(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def getmem_wg(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_nbi_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_nbi_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_wave(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_wg(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_nbi(dest, source, bytes, pe, qp_id=0):
    ...


@_shmem_module.dispatch
def putmem_nbi_wave(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_nbi_wg(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_wave(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_wg(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_nbi_wave(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_nbi_wg(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def signal_op(sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def signal_wait_until(sig_addr, cmp_, cmp_val):
    ...


# def wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
#     ...


# DON'T USE THIS. NVSHMEM 3.2.5 does not implement this
@_shmem_module.dispatch
def broadcastmem(team, dest, source, nelems, pe_root):
    ...


@_shmem_module.dispatch
def broadcastmem_warp(team, dest, source, nelems, pe_root):
    ...


@_shmem_module.dispatch
def broadcastmem_block(team, dest, source, nelems, pe_root):
    ...


@_shmem_module.dispatch
def broadcast(team, dest, source, nelems, pe_root):
    ...


@_shmem_module.dispatch
def broadcast_warp(team, dest, source, nelems, pe_root):
    ...


@_shmem_module.dispatch
def broadcast_block(team, dest, source, nelems, pe_root):
    ...


# DON'T USE THIS. NVSHMEM 3.2.5 does not implement this
@_shmem_module.dispatch
def fcollectmem(team, dest, source, nelems):
    ...


@_shmem_module.dispatch
def fcollectmem_warp(team, dest, source, nelems):
    ...


@_shmem_module.dispatch
def fcollectmem_block(team, dest, source, nelems):
    ...


@_shmem_module.dispatch
def fcollect(team, dest, source, nelems):
    ...


@_shmem_module.dispatch
def fcollect_warp(team, dest, source, nelems):
    ...


@_shmem_module.dispatch
def fcollect_block(team, dest, source, nelems):
    ...


### putmem_rma_* and putmem_signal_rma_* requires nvshmemi_* APIs. and you have to compile the .bc file yourself with shmem/nvshmem_bind/nvshmemi/build_nvshmemi_bc.sh
@_shmem_module.dispatch
def putmem_rma(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_rma_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_rma_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_rma_nbi(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_rma_nbi_block(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_rma_nbi_warp(dest, source, bytes, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


@_shmem_module.dispatch
def putmem_signal_rma_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


# TEAM translate
@_shmem_module.dispatch
def team_translate_pe(src_team, pe_in_src_team, dest_team):
    ...


# class nvshmemi_cmp_type(Enum):
NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_LE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_GE = 5
NVSHMEM_CMP_SENTINEL = sys.maxsize

# class nvshmemi_amo_t(Enum):
NVSHMEMI_AMO_ACK = 1
NVSHMEMI_AMO_INC = 2
NVSHMEMI_AMO_SET = 3
NVSHMEMI_AMO_ADD = 4
NVSHMEMI_AMO_AND = 5
NVSHMEMI_AMO_OR = 6
NVSHMEMI_AMO_XOR = 7
NVSHMEMI_AMO_SIGNAL = 8
NVSHMEM_SIGNAL_SET = 9
NVSHMEM_SIGNAL_ADD = 10
NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET  # Note - NVSHMEM_SIGNAL_SET == 9
NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD  # Note - NVSHMEM_SIGNAL_ADD == 10
NVSHMEMI_AMO_END_OF_NONFETCH = 11  # end of nonfetch atomics
NVSHMEMI_AMO_FETCH = 12
NVSHMEMI_AMO_FETCH_INC = 13
NVSHMEMI_AMO_FETCH_ADD = 14
NVSHMEMI_AMO_FETCH_AND = 15
NVSHMEMI_AMO_FETCH_OR = 16
NVSHMEMI_AMO_FETCH_XOR = 17
NVSHMEMI_AMO_SWAP = 18
NVSHMEMI_AMO_COMPARE_SWAP = 19
NVSHMEMI_AMO_OP_SENTINEL = sys.maxsize

# team node
NVSHMEM_TEAM_INVALID = -1
NVSHMEM_TEAM_WORLD = 0
NVSHMEM_TEAM_WORLD_INDEX = 0
NVSHMEM_TEAM_SHARED = 1
NVSHMEM_TEAM_SHARED_INDEX = 1
NVSHMEMX_TEAM_NODE = 2
NVSHMEM_TEAM_NODE_INDEX = 2
NVSHMEMX_TEAM_SAME_MYPE_NODE = 3
NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
NVSHMEMI_TEAM_SAME_GPU = 4
NVSHMEM_TEAM_SAME_GPU_INDEX = 4
NVSHMEMI_TEAM_GPU_LEADERS = 5
NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5
NVSHMEM_TEAMS_MIN = 6
NVSHMEM_TEAM_INDEX_MAX = sys.maxsize

# ROCSHMEM_CMPS (enum)
ROCSHMEM_CMP_EQ = 0
ROCSHMEM_CMP_NE = 1
ROCSHMEM_CMP_GT = 2
ROCSHMEM_CMP_GE = 3
ROCSHMEM_CMP_LT = 4
ROCSHMEM_CMP_LE = 5

# ROCSHMEM_SIGNAL_OPS (enum)
ROCSHMEM_SIGNAL_SET = 0
ROCSHMEM_SIGNAL_ADD = 1

# MoRI SHMEM atomicType (enum) - Not all types are currently supported.
MORI_AMO_ACK = 1
MORI_AMO_INC = 2
MORI_AMO_SET = 3
MORI_AMO_ADD = 4
MORI_AMO_AND = 5
MORI_AMO_OR = 6
MORI_AMO_XOR = 7
MORI_AMO_SIGNAL = 8
MORI_SIGNAL_SET = 9
MORI_SIGNAL_ADD = 10
MORI_AMO_SIGNAL_SET = MORI_SIGNAL_SET
MORI_AMO_SIGNAL_ADD = MORI_SIGNAL_ADD
MORI_AMO_END_OF_NONFETCH = 13
MORI_AMO_FETCH = 14
MORI_AMO_FETCH_INC = 15
MORI_AMO_FETCH_ADD = 16
MORI_AMO_FETCH_AND = 17
MORI_AMO_FETCH_OR = 18
MORI_AMO_FETCH_XOR = 19
MORI_AMO_SWAP = 20
MORI_AMO_COMPARE_SWAP = 21
MORI_AMO_OP_SENTINEL = sys.maxsize

