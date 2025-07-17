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
import datetime
import functools
import os

import pytest
import nvshmem.bindings
import nvshmem.core
import torch
import torch.distributed

import triton
import triton.backends
import triton.backends.nvidia
import triton.backends.nvidia.compiler
import triton.language as tl
from triton.language.extra.cuda.language_extra import (__syncthreads, load_v4_u32, multimem_st_b32, multimem_st_v2,
                                                       multimem_st_v4, ntid, st, tid, multimem_st_p_b32)
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, has_nvshmemi_bc_built, init_nvshmem_by_torch_process_group,
                               nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor,
                               is_nvshmem_multimem_supported)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def conditional_execution(condition_func):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func():
                return func(*args, **kwargs)
            else:
                print(f"{condition_func.__name__} not satisfied. skip {func.__name__}...")
                return None

        return wrapper

    return decorator


def test_nvshmem_basic():

    @triton.jit
    def _nvshmem_basic(output):
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            st(output, libshmem_device.my_pe())
            output += 1
            st(output, libshmem_device.team_my_pe(libshmem_device.NVSHMEM_TEAM_WORLD))
            output += 1
            st(output, libshmem_device.team_my_pe(libshmem_device.NVSHMEMX_TEAM_NODE))
            output += 1

            st(output, libshmem_device.n_pes())
            output += 1
            st(output, libshmem_device.team_n_pes(libshmem_device.NVSHMEM_TEAM_WORLD))
            output += 1
            st(output, libshmem_device.team_n_pes(libshmem_device.NVSHMEMX_TEAM_NODE))

    print("nvshmem basic start...")
    output = nvshmem_create_tensor((6, ), torch.int32)
    _nvshmem_basic[(1, )](output)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    try:
        torch.testing.assert_close(
            output,
            torch.tensor([RANK, RANK, LOCAL_RANK, WORLD_SIZE, WORLD_SIZE, LOCAL_WORLD_SIZE], dtype=torch.int32,
                         device="cuda")), output
    except Exception as e:
        print(" ❌ nvshmem basic failed")
        raise (e)
    else:
        print("✅ nvshmem basic pass")

    nvshmem_free_tensor_sync(output)


def test_nvshmemx_getmem_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemx_getmem(ptr, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.getmem_nbi_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_nbi_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.getmem_nbi(
                            ptr + pid * bytes_per_rank + thread_idx,
                            ptr + pid * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.getmem_block(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.getmem_warp(
                        ptr + pid * bytes_per_rank,
                        ptr + pid * bytes_per_rank,
                        bytes_per_rank,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < bytes_per_rank:
                        libshmem_device.getmem(
                            ptr + pid * bytes_per_rank + thread_idx,
                            ptr + pid * bytes_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_getmem_block",
                ("warp", False): "nvshmemx_getmem_warp",
                ("thread", False): "nvshmem_getmem",
                ("block", True): "nvshmemx_getmem_nbi_block",
                ("warp", True): "nvshmemx_getmem_nbi_warp",
                ("thread", True): "nvshmem_getmem_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemx_getmem[(WORLD_SIZE, )](
                t,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=1 if scope == "warp" else 4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")

    nvshmem_free_tensor_sync(t)


def test_nvshmemx_putmem_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemx_putmem(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_nbi(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_putmem_block",
                ("warp", False): "nvshmemx_putmem_warp",
                ("thread", False): "nvshmem_putmem",
                ("block", True): "nvshmemx_putmem_nbi_block",
                ("warp", True): "nvshmemx_putmem_nbi_warp",
                ("thread", True): "nvshmem_putmem_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemx_putmem[(WORLD_SIZE, )](
                t,
                N // WORLD_SIZE,
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(t)


def test_nvshmem_signal():

    @triton.jit
    def _pingpong(t, iters):
        # pingpong for rank 0-1, 2-3, ...
        mype = libshmem_device.my_pe()
        thread_idx = tid(axis=0)
        if thread_idx == 0:
            for n in range(iters):
                if mype == 0:
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        1,
                    )
                elif mype == 1:
                    libshmem_device.signal_op(
                        t,
                        1 + n,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        0,
                    )
                    libshmem_device.signal_wait_until(t, libshmem_device.NVSHMEM_CMP_EQ, 1 + n)
        __syncthreads()

    print("test nvshmemx_signal with pingpong...")
    t = nvshmem_create_tensor((1, ), NVSHMEM_SIGNAL_DTYPE)
    t.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    _pingpong[(1, )](t, 100, num_warps=1)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    if nvshmem.bindings.nvshmem.my_pe() == 0:
        try:
            torch.testing.assert_close(t.to(torch.int32), torch.ones([1], dtype=torch.int32, device="cuda") * 100)
        except Exception as e:
            print("❌ nvshmemx_signal with pingpong failed")
            raise e
        else:
            print("✅ nvshmemx_signal with pingpong pass")
    nvshmem_free_tensor_sync(t)


def test_nvshmemx_putmem_signal_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemx_putmem_signal(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_signal_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_nbi_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal_nbi(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_signal_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)
    signal = nvshmem_create_tensor((WORLD_SIZE, ), NVSHMEM_SIGNAL_DTYPE)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemx_putmem_signal_block",
                ("warp", False): "nvshmemx_putmem_signal_warp",
                ("thread", False): "nvshmem_putmem_signal",
                ("block", True): "nvshmemx_putmem_signal_nbi_block",
                ("warp", True): "nvshmemx_putmem_signal_nbi_warp",
                ("thread", True): "nvshmem_putmem_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemx_putmem_signal[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=NVSHMEM_SIGNAL_DTYPE,
                                                              device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(t)
    nvshmem_free_tensor_sync(signal)


def test_nvshmem_barrier_sync_quiet_fence():
    """only test runs, no result checked"""

    @triton.jit
    def _nvshmem_barrier_sync_quiet_fence():
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_all_block()
            libshmem_device.sync_all_block()

            if thread_idx / 32 == 0:
                libshmem_device.barrier_all_warp()
                libshmem_device.sync_all_warp()

            if thread_idx == 0:
                libshmem_device.barrier_all()
                libshmem_device.sync_all()

        libshmem_device.quiet()
        libshmem_device.fence()

    @triton.jit
    def _nvshmem_barrier_sync_quiet_fence_with_team(team):
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid == 0:
            libshmem_device.barrier_block(team)
            libshmem_device.team_sync_block(team)

            if thread_idx / 32 == 0:
                libshmem_device.barrier_warp(team)
                libshmem_device.team_sync_warp(team)

            if thread_idx == 0:
                libshmem_device.barrier(team)

    print("test nvshmem_barrier/nvshmem_sync/nvshmem_quiet/nvshmem_fence all in one...")
    _nvshmem_barrier_sync_quiet_fence[(1, )](num_warps=4)
    torch.cuda.synchronize()
    print("✅ nvshmem_barrier_all/nvshmem_sync/nvshmem_quiet/nvshmem_fence pased...")
    _nvshmem_barrier_sync_quiet_fence_with_team[(1, )](nvshmem.core.Teams.TEAM_NODE, num_warps=4)
    torch.cuda.synchronize()
    print("✅ nvshmem_barrier/nvshmemx_team_sync pased...")


def test_nvshmem_broadcast(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmem_broadcast(dst, src, nbytes, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if scope == "block":
            libshmem_device.broadcast_block(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
        if scope == "warp":
            if wid == 0:
                libshmem_device.broadcast_warp(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
                __syncthreads()
        if scope == "thread":
            if thread_idx == 0:
                libshmem_device.broadcast(libshmem_device.NVSHMEM_TEAM_WORLD, dst, src, nbytes, 0)
            __syncthreads()

    src = nvshmem_create_tensor((N, ), dtype)
    dst = nvshmem_create_tensor((N, ), dtype)
    for scope in ["block", "warp", "thread"]:
        api = {
            "block": "nvshmemx_broadcast_block",
            "warp": "nvshmemx_broadcast_warp",
            "thread": "nvshmem_broadcast",
        }[scope]
        print(f"running {api}...")
        src.fill_(RANK + 1)
        dst.fill_(-1)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _nvshmem_broadcast[(1, )](
            dst,
            src,
            src.nbytes,
            scope,
            num_warps=4,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        t_expected = torch.ones_like(dst)
        try:
            torch.testing.assert_close(dst, t_expected)
        except Exception as e:
            print(f" ❌ {api} failed")
            print(dst)
            raise (e)
        else:
            print(f"✅ {api} pass")

    nvshmem_free_tensor_sync(src)
    nvshmem_free_tensor_sync(dst)


def test_nvshmem_fcollect(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmem_fcollect(dst, src, nbytes, scope: tl.constexpr):
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if scope == "block":
            libshmem_device.fcollect_block(
                libshmem_device.NVSHMEM_TEAM_WORLD,
                dst,
                src,
                nbytes,
            )
        if scope == "warp":
            if wid == 0:
                libshmem_device.fcollect_warp(
                    libshmem_device.NVSHMEM_TEAM_WORLD,
                    dst,
                    src,
                    nbytes,
                )
            __syncthreads()
        if scope == "thread":
            if thread_idx == 0:
                libshmem_device.fcollect(
                    libshmem_device.NVSHMEM_TEAM_WORLD,
                    dst,
                    src,
                    nbytes,
                )
            __syncthreads()

    src = nvshmem_create_tensor((N, ), dtype)
    dst = nvshmem_create_tensor((N * WORLD_SIZE, ), dtype)
    for scope in ["block", "warp", "thread"]:
        api = {
            "block": "nvshmemx_fcollect_block",
            "warp": "nvshmemx_fcollect_warp",
            "thread": "nvshmem_fcollect",
        }[scope]
        print(f"running {api}...")
        src.fill_(RANK + 1)
        dst.fill_(-1)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        _nvshmem_fcollect[(1, )](
            dst,
            src,
            src.nbytes // src.itemsize,
            scope,
            num_warps=4,
        )
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        t_expected = (torch.ones_like(dst).reshape(
            (WORLD_SIZE, -1)) * torch.arange(1, 1 + WORLD_SIZE, device="cuda").to(dtype)[:, None]).flatten()
        try:
            torch.testing.assert_close(dst, t_expected)
        except Exception as e:
            print(f" ❌ {api} failed")
            print(dst)
            raise (e)
        else:
            print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(src)
    nvshmem_free_tensor_sync(dst)


@conditional_execution(is_nvshmem_multimem_supported)
def test_nvshmem_multimem_st(N):

    @triton.jit
    def _nvshmem_multimem_st_v4(symm_ptr, nbytes):
        """ TODO(houqi.1993) it's expected that multimem.st.v3.fp32 is supported. but actually no. ptxas won't compile:
        https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem """
        thread_idx = tid(axis=0)
        block_dim = ntid(axis=0)
        pid = tl.program_id(0)
        npid = tl.num_programs(0)
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
        step = 128 // symm_ptr.dtype.element_ty.primitive_bitwidth  # 128 bits = 16 bytes
        for n in range(thread_idx + block_dim * pid, nbytes // 16, block_dim * npid):
            val0, val1, val2, val3 = load_v4_u32(symm_ptr + n * step)
            multimem_st_v4(mc_ptr + n * step, val0, val1, val2, val3)

    @triton.jit
    def _nvshmem_multimem_st_v2(symm_ptr, nbytes):
        thread_idx = tid(axis=0)
        block_dim = ntid(axis=0)
        pid = tl.program_id(0)
        npid = tl.num_programs(0)
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
        step = 128 // symm_ptr.dtype.element_ty.primitive_bitwidth  # 128 bits = 16 bytes
        for n in range(thread_idx + block_dim * pid, nbytes // 16, block_dim * npid):
            val0, val1, val2, val3 = load_v4_u32(symm_ptr + n * step)
            multimem_st_v2(mc_ptr + n * step, val0, val1)
            multimem_st_v2(mc_ptr + n * step + step // 2, val2, val3)

    @triton.jit
    def _nvshmem_multimem_st_b32(symm_ptr, nbytes):
        thread_idx = tid(axis=0)
        block_dim = ntid(axis=0)
        pid = tl.program_id(0)
        npid = tl.num_programs(0)
        symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
        mc_ptr = tl.cast(mc_ptr, tl.pointer_type(tl.int8))
        for n in range(thread_idx + block_dim * pid, nbytes // 16, block_dim * npid):
            val0, val1, val2, val3 = load_v4_u32(symm_ptr + n * 16)
            multimem_st_b32(mc_ptr + n * 16, val0)
            multimem_st_b32(mc_ptr + n * 16 + 4, val1)
            multimem_st_b32(mc_ptr + n * 16 + 8, val2)
            multimem_st_b32(mc_ptr + n * 16 + 12, val3)

    @triton.jit
    def _nvshmem_multimem_st_p_b32(symm_ptr, val):
        thread_idx = tid(axis=0)
        pid = tl.program_id(0)
        symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))
        mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_ptr)
        if thread_idx == 0 and pid == 0:
            # should write not data with mask=0
            multimem_st_p_b32(mc_ptr, tl.cast(val, tl.uint32), 0)

    dtype = torch.float16
    t: torch.Tensor = nvshmem_create_tensor((N, ), dtype)
    t.fill_(1 + RANK)
    t_expected = torch.ones_like(t)
    # test multimem.st without .v2/v4
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    if RANK == 0:
        _nvshmem_multimem_st_b32[(4, )](t, t.nbytes, num_warps=4)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.testing.assert_close(t, t_expected)

    # TODO(houqi.1993) multimem.st should support v4/v2, but it is not. when ptxas fix it, we will support it.
    with pytest.raises(triton.runtime.errors.PTXASError, match=r"PTXAS error: Internal Triton PTX codegen error"):
        _nvshmem_multimem_st_v2.warmup(t, t.nbytes, grid=(4, ))
    print(f"✅ _nvshmem_multimem_st with {dtype} v2 compiled failed as expected")
    with pytest.raises(triton.runtime.errors.PTXASError, match=r"PTXAS error: Internal Triton PTX codegen error"):
        _nvshmem_multimem_st_v4.warmup(t, t.nbytes, grid=(4, ))
    print(f"✅ _nvshmem_multimem_st with {dtype} v4 compiled failed as expected")

    _nvshmem_multimem_st_p_b32[(1, )](t, 0xffffffff)
    if RANK == 0:  # RANK 0 fails may cause RANK 1 got an Exception. only check with 1 rank
        with pytest.raises(AssertionError, match=r"Tensor-likes are not close!"):
            # t is not changed as expected. but ptxas has a BUG here.
            torch.testing.assert_close(t, t_expected)

    nvshmem_free_tensor_sync(t)
    print(f"✅ _nvshmem_multimem_st with {dtype} done")


@conditional_execution(has_nvshmemi_bc_built)
def test_nvshmemi_putmem_rma(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemi_putmem_rma_kernel(
        ptr,
        elems_per_rank,
        scope: tl.constexpr,
        nbi: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
    ):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_rma_nbi_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_rma_nbi_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_rma_nbi(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_rma_block(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "warp":
                    libshmem_device.putmem_rma_warp(
                        ptr + mype * elems_per_rank,
                        ptr + mype * elems_per_rank,
                        elems_per_rank * ELEM_SIZE,
                        pid,
                    )
                elif scope == "thread":
                    if thread_idx < elems_per_rank:
                        libshmem_device.putmem_rma(
                            ptr + mype * elems_per_rank + thread_idx,
                            ptr + mype * elems_per_rank + thread_idx,
                            1,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemi_transfer_rma_put_block",
                ("warp", False): "nvshmemi_transfer_rma_put_warp",
                ("thread", False): "nvshmemi_transfer_rma",
                ("block", True): "nvshmemi_transfer_rma_put_nbi_block",
                ("warp", True): "nvshmemi_transfer_rma_put_nbi_warp",
                ("thread", True): "nvshmemi_transfer_rma_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemi_putmem_rma_kernel[(WORLD_SIZE, )](
                t,
                N // WORLD_SIZE,
                scope,
                nbi,
                ELEM_SIZE=dtype.itemsize,
                num_warps=1 if scope == "warp" else 4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                raise (e)
            else:
                print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(t)


@conditional_execution(has_nvshmemi_bc_built)
def test_nvshmemi_putmem_rma_signal_with_scope(N, dtype: torch.dtype = torch.int8):

    @triton.jit
    def _nvshmemi_putmem_rma_signal_kernel(ptr, signal, bytes_per_rank, scope: tl.constexpr, nbi: tl.constexpr):
        mype = libshmem_device.my_pe()
        pid = tl.program_id(axis=0)
        thread_idx = tid(axis=0)
        wid = thread_idx // 32
        if pid != mype:
            if nbi:
                if scope == "block":
                    libshmem_device.putmem_signal_rma_nbi_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_rma_nbi_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal_rma_nbi(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")
            else:
                if scope == "block":
                    libshmem_device.putmem_signal_rma_block(
                        ptr + mype * bytes_per_rank,
                        ptr + mype * bytes_per_rank,
                        bytes_per_rank,
                        signal + mype,
                        1,
                        libshmem_device.NVSHMEM_SIGNAL_SET,
                        pid,
                    )
                elif scope == "warp":
                    if wid == 0:
                        libshmem_device.putmem_signal_rma_warp(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                elif scope == "thread":
                    if thread_idx == 0:
                        libshmem_device.putmem_signal_rma(
                            ptr + mype * bytes_per_rank,
                            ptr + mype * bytes_per_rank,
                            bytes_per_rank,
                            signal + mype,
                            1,
                            libshmem_device.NVSHMEM_SIGNAL_SET,
                            pid,
                        )
                else:
                    raise ValueError("scope must be block, warp, or thread")

    t = nvshmem_create_tensor((N, ), dtype)
    signal = nvshmem_create_tensor((WORLD_SIZE, ), NVSHMEM_SIGNAL_DTYPE)

    for scope in ["block", "warp", "thread"]:
        for nbi in [True, False]:
            api = {
                ("block", False): "nvshmemi_transfer_put_signal_block",
                ("warp", False): "nvshmemi_transfer_put_signal_warp",
                ("thread", False): "nvshmemi_transfer_put_signal",
                ("block", True): "nvshmemi_transfer_put_signal_nbi_block",
                ("warp", True): "nvshmemi_transfer_put_signal_nbi_warp",
                ("thread", True): "nvshmemi_transfer_put_signal_nbi",
            }[(scope, nbi)]
            print(f"runing {api}...")
            t.fill_(RANK + 1)
            signal.fill_(0)
            signal[RANK].fill_(1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            _nvshmemi_putmem_rma_signal_kernel[(WORLD_SIZE, )](
                t,
                signal,
                t.nbytes // WORLD_SIZE,
                scope,
                nbi,
                num_warps=4,
            )
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            t_expected = (torch.arange(1, WORLD_SIZE + 1, dtype=dtype, device="cuda").reshape(
                (WORLD_SIZE, 1)).repeat(1, N // WORLD_SIZE).flatten())
            try:
                torch.testing.assert_close(t, t_expected)
                torch.testing.assert_close(signal, torch.ones((WORLD_SIZE, ), dtype=NVSHMEM_SIGNAL_DTYPE,
                                                              device="cuda"))
            except Exception as e:
                print(f" ❌ {api} failed")
                print(t.reshape(WORLD_SIZE, -1))
                print(signal)
                raise (e)
            else:
                print(f"✅ {api} pass")
    nvshmem_free_tensor_sync(t)
    nvshmem_free_tensor_sync(signal)


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    init_nvshmem_by_torch_process_group(TP_GROUP)

    test_nvshmem_basic()
    test_nvshmemx_getmem_with_scope(31 * WORLD_SIZE, torch.int8)
    test_nvshmemx_putmem_with_scope(16 * WORLD_SIZE, torch.int8)
    test_nvshmemx_putmem_signal_with_scope(20 * WORLD_SIZE, torch.int8)
    test_nvshmem_signal()
    test_nvshmem_barrier_sync_quiet_fence()
    test_nvshmem_broadcast(32 * WORLD_SIZE, torch.int8)

    # some ranks hangs. don't know why
    # test_nvshmem_fcollect(1024, torch.int8)
    test_nvshmem_multimem_st(1024)

    test_nvshmemi_putmem_rma(16 * WORLD_SIZE, torch.int8)
    test_nvshmemi_putmem_rma_signal_with_scope(16 * WORLD_SIZE, torch.int8)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
