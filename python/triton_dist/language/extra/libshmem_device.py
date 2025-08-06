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
import sys


def my_pe():
    ...


def n_pes():
    ...


def team_my_pe(team):
    ...


def team_n_pes(team):
    ...


def int_p(dest, value, pe):
    ...


def remote_ptr(local_ptr, pe):
    ...


def remote_mc_ptr(team, ptr):
    ...


def barrier_all():
    ...


def barrier_all_block():
    ...


def barrier_all_warp():
    ...


def barrier(team):
    ...


def barrier_block(team):
    ...


def barrier_warp(team):
    ...


def sync_all():
    ...


def sync_all_block():
    ...


def sync_all_warp():
    ...


def team_sync_block(team):
    ...


def team_sync_warp(team):
    ...


def quiet():
    ...


def fence():
    ...


def getmem_nbi_block(dest, source, bytes, pe):
    ...


def getmem_block(dest, source, bytes, pe):
    ...


def getmem_nbi_warp(dest, source, bytes, pe):
    ...


def getmem_warp(dest, source, bytes, pe):
    ...


def getmem_nbi(dest, source, bytes, pe):
    ...


def getmem(dest, source, bytes, pe):
    ...


def putmem_block(dest, source, bytes, pe):
    ...


def putmem_nbi_block(dest, source, bytes, pe):
    ...


def putmem_warp(dest, source, bytes, pe):
    ...


def putmem_nbi_warp(dest, source, bytes, pe):
    ...


def putmem(dest, source, bytes, pe):
    ...


def putmem_nbi(dest, source, bytes, pe):
    ...


def putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    ...


def signal_op(sig_addr, signal, sig_op, pe):
    ...


def signal_wait_until(sig_addr, cmp_, cmp_val):
    ...


# DON'T USE THIS. NVSHMEM 3.2.5 does not implement this
def broadcastmem(team, dest, source, nelems, pe_root):
    ...


def broadcastmem_warp(team, dest, source, nelems, pe_root):
    ...


def broadcastmem_block(team, dest, source, nelems, pe_root):
    ...


def broadcast(team, dest, source, nelems, pe_root):
    ...


def broadcast_warp(team, dest, source, nelems, pe_root):
    ...


def broadcast_block(team, dest, source, nelems, pe_root):
    ...


# DON'T USE THIS. NVSHMEM 3.2.5 does not implement this
def fcollectmem(team, dest, source, nelems):
    ...


def fcollectmem_warp(team, dest, source, nelems):
    ...


def fcollectmem_block(team, dest, source, nelems):
    ...


def fcollect(team, dest, source, nelems):
    ...


def fcollect_warp(team, dest, source, nelems):
    ...


def fcollect_block(team, dest, source, nelems):
    ...


### putmem_rma_* and putmem_signal_rma_* requires nvshmemi_* APIs. and you have to compile the .bc file yourself with shmem/nvshmem_bind/nvshmemi/build_nvshmemi_bc.sh
def putmem_rma(dest, source, bytes, pe):
    ...


def putmem_rma_block(dest, source, bytes, pe):
    ...


def putmem_rma_warp(dest, source, bytes, pe):
    ...


def putmem_rma_nbi(dest, source, bytes, pe):
    ...


def putmem_rma_nbi_block(dest, source, bytes, pe):
    ...


def putmem_rma_nbi_warp(dest, source, bytes, pe):
    ...


def putmem_signal_rma(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


def putmem_signal_rma_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


def putmem_signal_rma_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


def putmem_signal_rma_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


def putmem_signal_rma_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


def putmem_signal_rma_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe):
    pass


# TEAM translate
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

## TODO: add rocshmem
