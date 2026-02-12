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
Multi-GPU FlashComm dispatch test using torchrun (one process per GPU).
Uses raw cudaMalloc + cudaIpc* for cross-process GPU memory sharing.

NOTE: PyTorch 2.9's CUDA VMM allocator produces IPC-mapped addresses
that are NOT kernel-accessible via P2P. We must use raw cudaMalloc
for cross-GPU buffers to ensure proper P2P accessibility.

Usage:
  bash launch.sh --nproc_per_node=2 design/test_flashcomm_torchrun.py
  bash launch.sh --nproc_per_node=4 design/test_flashcomm_torchrun.py
"""

import os
import sys
import time
import ctypes
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist

from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

from flashcomm_compute import (
    kernel_compute_offset,
    kernel_compute_dispatch_layout,
    NUM_WARPS,
    MAX_EXPERTS_PLUS_1,
    BLOCK_SIZE,
)
from flashcomm_dispatch import (
    kernel_dispatch_intranode_v1,
    TOTAL_SMEM as DISPATCH_SMEM,
    V1_TOTAL_THREADS,
)
from flashcomm_barrier import (
    kernel_barrier_all_on_stream,
    NUM_THREADS as BARRIER_THREADS,
)

# ============================================================================
# CUDA runtime bindings for raw memory allocation and IPC
# ============================================================================
_rt = ctypes.CDLL('libcudart.so')
_rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_rt.cudaMalloc.restype = ctypes.c_int
_rt.cudaFree.argtypes = [ctypes.c_void_p]
_rt.cudaFree.restype = ctypes.c_int
_rt.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_rt.cudaMemset.restype = ctypes.c_int
_rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_rt.cudaMemcpy.restype = ctypes.c_int
_rt.cudaDeviceSynchronize.restype = ctypes.c_int
_rt.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
_rt.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
_rt.cudaGetLastError.argtypes = []
_rt.cudaGetLastError.restype = ctypes.c_int

IPC_HANDLE_SIZE = 64


class _IpcHandle(ctypes.Structure):
    _fields_ = [('data', ctypes.c_ubyte * IPC_HANDLE_SIZE)]


_ipc_rt = ctypes.CDLL('libcudart.so')
_ipc_rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
_ipc_rt.cudaIpcGetMemHandle.restype = ctypes.c_int
_ipc_rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
_ipc_rt.cudaIpcOpenMemHandle.restype = ctypes.c_int
_ipc_rt.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
_ipc_rt.cudaIpcCloseMemHandle.restype = ctypes.c_int


class RawCudaBuffer:
    """Raw cudaMalloc buffer for P2P-accessible cross-GPU memory."""

    def __init__(self, size_bytes):
        self.size_bytes = size_bytes
        self.ptr = ctypes.c_void_p()
        err = _rt.cudaMalloc(ctypes.byref(self.ptr), size_bytes)
        assert err == 0, f"cudaMalloc failed: err={err}"
        _rt.cudaMemset(self.ptr, 0, size_bytes)

    def data_ptr(self):
        return self.ptr.value

    def zero_(self):
        _rt.cudaMemset(self.ptr, 0, self.size_bytes)

    def to_torch(self, dtype, shape):
        """Copy to a PyTorch tensor for inspection."""
        t = torch.zeros(shape, dtype=dtype, device=f'cuda:{torch.cuda.current_device()}')
        _rt.cudaMemcpy(ctypes.c_void_p(t.data_ptr()), self.ptr, self.size_bytes, 3)  # D2D
        return t

    def free(self):
        if self.ptr.value:
            _rt.cudaFree(self.ptr)
            self.ptr = ctypes.c_void_p()

    def __del__(self):
        self.free()


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def enable_peer_access_for(my_gpu, all_gpus):
    for g in all_gpus:
        if g != my_gpu:
            err = _rt.cudaDeviceEnablePeerAccess(g, 0)
            if err == 704:
                _rt.cudaGetLastError()


def ipc_share_buffer(local_buf, world_size, rank):
    """Share a RawCudaBuffer across processes via CUDA IPC.
    Uses int-array serialization to avoid null-byte truncation.
    Returns list[world_size] of data_ptr values usable in this process.
    """
    _rt.cudaDeviceSynchronize()

    # Get IPC handle and serialize as int array
    handle = _IpcHandle()
    err = _ipc_rt.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(local_buf.data_ptr()))
    assert err == 0, f"cudaIpcGetMemHandle failed: err={err}"
    handle_ints = list(handle.data)

    # Exchange handles
    all_handles = [None] * world_size
    dist.all_gather_object(all_handles, handle_ints)
    dist.barrier()

    # Open remote handles
    ptrs = []
    remote_handles = []  # keep references alive
    for r in range(world_size):
        if r == rank:
            ptrs.append(local_buf.data_ptr())
        else:
            h = _IpcHandle()
            for i in range(IPC_HANDLE_SIZE):
                h.data[i] = all_handles[r][i]
            p = ctypes.c_void_p()
            err = _ipc_rt.cudaIpcOpenMemHandle(ctypes.byref(p), h, 1)  # cudaIpcMemLazyEnablePeerAccess
            assert err == 0, f"cudaIpcOpenMemHandle(rank={r}) failed: err={err}"
            ptrs.append(p.value)
            remote_handles.append(p)

    return ptrs, remote_handles


def make_ptr_tensor(ptrs, device):
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def build_kernel_on_gpu(gpu_id, build_fn):
    torch.cuda.set_device(gpu_id)
    k = build_fn()
    _ = k.launcher
    return k


def reference_global_rank(flat_idx, num_experts):
    M = flat_idx.numel()
    perm = torch.argsort(flat_idx, stable=True)
    k = flat_idx[perm]
    i = torch.arange(M, device=flat_idx.device, dtype=torch.int64)
    head = torch.empty(M, device=flat_idx.device, dtype=torch.bool)
    head[0] = True
    head[1:] = k[1:] != k[:-1]
    boundary = torch.where(head, i, torch.full_like(i, -1))
    run_start = torch.cummax(boundary, dim=0).values
    rank_sorted = (i - run_start).to(torch.int32)
    out = torch.empty(M, device=flat_idx.device, dtype=torch.int32)
    out[perm] = rank_sorted
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-token", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_id = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    num_experts = args.num_experts
    topk = args.topk
    num_token = args.num_token
    hidden_size = args.hidden
    experts_per_rank = num_experts // world_size
    max_recv = num_token * world_size * topk + 1024

    log(rank, "=" * 70)
    log(rank, f"FlashComm torchrun Test: {world_size} ranks")
    log(rank, f"  experts={num_experts}, topk={topk}, tokens={num_token}, hidden={hidden_size}")
    log(rank, "=" * 70)

    enable_peer_access_for(gpu_id, list(range(world_size)))
    dist.barrier()

    # Build kernels
    log(rank, "  Building kernels ...")
    t0 = time.time()
    ep1 = max(num_experts + 1, MAX_EXPERTS_PLUS_1)
    compute_offset_k = build_kernel_on_gpu(
        gpu_id, lambda: kernel_compute_offset.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(4, 1, 1),
                                                    block=(BLOCK_SIZE, 1, 1), shared_mem_bytes=NUM_WARPS * ep1 * 4))
    dispatch_layout_k = build_kernel_on_gpu(
        gpu_id, lambda: kernel_compute_dispatch_layout.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(
            4, 1, 1), block=(BLOCK_SIZE, 1, 1), shared_mem_bytes=NUM_WARPS * 4))
    dispatch_v1_k = build_kernel_on_gpu(
        gpu_id, lambda: kernel_dispatch_intranode_v1.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(
            128, 1, 1), block=(V1_TOTAL_THREADS, 1, 1), shared_mem_bytes=DISPATCH_SMEM + 4096))
    barrier_k = build_kernel_on_gpu(
        gpu_id, lambda: kernel_barrier_all_on_stream.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(
            1, 1, 1), block=(BARRIER_THREADS, 1, 1), shared_mem_bytes=0))
    dist.barrier()
    log(rank, f"    Built in {time.time()-t0:.1f}s")

    # Input data
    torch.manual_seed(args.seed + rank)
    topk_indices = torch.randint(0, num_experts + 1, (num_token, topk), device=device, dtype=torch.int32)
    x_data = ((torch.arange(num_token, device=device, dtype=torch.float32).unsqueeze(1) + rank * 1000) * 0.01).expand(
        -1, hidden_size).contiguous().to(torch.bfloat16)
    weights = torch.rand(num_token, topk, device=device, dtype=torch.float32)

    # Cross-GPU buffers: raw cudaMalloc for P2P accessibility
    # (PyTorch's CUDA VMM allocator produces IPC addresses not kernel-accessible)
    recv_x_buf = RawCudaBuffer(max_recv * hidden_size * 2)  # bf16
    recv_w_buf = RawCudaBuffer(max_recv * topk * 4)  # f32
    recv_s_buf = RawCudaBuffer(max_recv * topk * 4)  # i32
    barrier_buf = RawCudaBuffer(world_size * 4)  # i32
    full_splits_buf = RawCudaBuffer(world_size * (num_experts + 1) * 4)  # i32
    _rt.cudaDeviceSynchronize()
    dist.barrier()

    # Exchange IPC handles via raw CUDA IPC
    log(rank, "  Sharing buffers via raw CUDA IPC ...")
    rx_ptrs, _rx_h = ipc_share_buffer(recv_x_buf, world_size, rank)
    rw_ptrs, _rw_h = ipc_share_buffer(recv_w_buf, world_size, rank)
    rs_ptrs, _rs_h = ipc_share_buffer(recv_s_buf, world_size, rank)
    bar_ptrs, _bar_h = ipc_share_buffer(barrier_buf, world_size, rank)
    fs_ptrs, _fs_h = ipc_share_buffer(full_splits_buf, world_size, rank)
    dist.barrier()
    log(rank, "    Done")

    rx_ptr_t = make_ptr_tensor(rx_ptrs, device)
    rw_ptr_t = make_ptr_tensor(rw_ptrs, device)
    rs_ptr_t = make_ptr_tensor(rs_ptrs, device)
    bar_ptr_t = make_ptr_tensor(bar_ptrs, device)
    fs_ptr_t = make_ptr_tensor(fs_ptrs, device)

    # Phase 1a: compute_offset
    log(rank, "  Phase 1a: compute_offset ...")
    t1 = time.time()
    nte = num_token * topk
    ntiles = (nte + BLOCK_SIZE - 1) // BLOCK_SIZE
    bch = torch.zeros((ntiles, num_experts + 1), device=device, dtype=torch.int32)
    tok_off = torch.zeros((num_token, topk), device=device, dtype=torch.int32)
    exp_cnt = torch.zeros((num_experts + 1, ), device=device, dtype=torch.int32)
    compute_offset_k(topk_indices, num_token, topk, num_experts, bch, tok_off, exp_cnt)
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, f"    Done ({time.time()-t1:.2f}s)")

    # Phase 1b: dispatch_layout (cross-rank barrier)
    log(rank, "  Phase 1b: dispatch_layout ...")
    t1b = time.time()
    rbo = torch.zeros(world_size * experts_per_rank * world_size, device=device, dtype=torch.int32)
    scat = torch.full((num_token, topk), -1, device=device, dtype=torch.int32)
    smask = torch.zeros((num_token, topk), device=device, dtype=torch.int32)
    rtc = torch.zeros(world_size, device=device, dtype=torch.int32)
    dispatch_layout_k(topk_indices, tok_off, exp_cnt, fs_ptr_t, bar_ptr_t, rbo, scat, smask, 0, rtc, num_token, topk,
                      num_experts, rank, world_size)
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, f"    Done ({time.time()-t1b:.2f}s)")

    # Print stats
    info = torch.tensor([smask.sum().item(), (scat >= 0).sum().item()], dtype=torch.int64, device=device)
    info_all = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(info_all, info)
    rtc_all = [torch.zeros(world_size, dtype=torch.int32, device=device) for _ in range(world_size)]
    dist.all_gather(rtc_all, rtc)
    if rank == 0:
        for r in range(world_size):
            print(
                f"    Rank {r}: recv={rtc_all[r].cpu().tolist()}, sm={info_all[r][0].item()}, vs={info_all[r][1].item()}",
                flush=True)

    # Reset barrier
    barrier_buf.zero_()
    _rt.cudaDeviceSynchronize()
    dist.barrier()

    # Phase 2: dispatch
    log(rank, "  Phase 2: dispatch_intranode_v1 ...")
    t2 = time.time()
    dispatch_v1_k(x_data, smask, weights, topk_indices, scat, num_token, hidden_size, experts_per_rank, rank,
                  world_size, rx_ptr_t, rw_ptr_t, rs_ptr_t)
    torch.cuda.synchronize()
    dist.barrier()
    rxd = recv_x_buf.to_torch(dtype=torch.bfloat16, shape=(max_recv, hidden_size))
    nz = (rxd != 0).any(dim=1).sum().item()
    nz_t = torch.tensor([nz], dtype=torch.int64, device=device)
    nz_all = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(nz_all, nz_t)
    log(rank, f"    Done ({time.time()-t2:.2f}s)")
    if rank == 0:
        for r in range(world_size):
            print(f"    Rank {r} recv_x: {nz_all[r].item()} non-zero rows", flush=True)

    # Phase 3: barrier
    log(rank, "  Phase 3: barrier ...")
    barrier_buf.zero_()
    _rt.cudaDeviceSynchronize()
    dist.barrier()
    t3 = time.time()
    barrier_k(bar_ptr_t, rank, world_size)
    torch.cuda.synchronize()
    dist.barrier()
    log(rank, f"    Done ({time.time()-t3:.2f}s)")

    # Verification
    log(rank, "  Verifying ...")
    g_topk = [torch.zeros_like(topk_indices) for _ in range(world_size)]
    g_x = [torch.zeros_like(x_data) for _ in range(world_size)]
    g_sm = [torch.zeros_like(smask) for _ in range(world_size)]
    g_sc = [torch.zeros_like(scat) for _ in range(world_size)]
    dist.all_gather(g_topk, topk_indices)
    dist.all_gather(g_x, x_data)
    dist.all_gather(g_sm, smask)
    dist.all_gather(g_sc, scat)

    my_rx = recv_x_buf.to_torch(dtype=torch.bfloat16, shape=(max_recv, hidden_size))
    g_rx = [torch.zeros(max_recv, hidden_size, dtype=torch.bfloat16, device=device) for _ in range(world_size)]
    dist.all_gather(g_rx, my_rx)
    g_rtc = [torch.zeros(world_size, dtype=torch.int32, device=device) for _ in range(world_size)]
    dist.all_gather(g_rtc, rtc)

    if rank == 0:
        ok = True
        cnts = []
        for r in range(world_size):
            cnts.append(torch.bincount(g_topk[r].flatten(), minlength=num_experts + 1).to(torch.int32))
        ref_rc = torch.zeros(world_size, dtype=torch.int32, device=device)
        for tgt in range(world_size):
            s = 0
            for src in range(world_size):
                for e in range(tgt * experts_per_rank, (tgt + 1) * experts_per_rank):
                    s += cnts[src][e].item()
            ref_rc[tgt] = s
        for r in range(world_size):
            exp_v = ref_rc[r].item()
            got_v = g_rtc[r][r].item()
            if got_v != exp_v:
                print(f"    FAIL: rank {r} recv_count: {got_v} vs {exp_v}", flush=True)
                ok = False
            else:
                print(f"    PASS: rank {r} recv_count = {got_v}", flush=True)

        tc = 0
        tm = 0
        for sr in range(world_size):
            for t in range(num_token):
                for k in range(topk):
                    if g_sm[sr][t, k].item() == 1:
                        ex = g_topk[sr][t, k].item()
                        if ex >= num_experts: continue
                        tr = ex // experts_per_rank
                        di = g_sc[sr][t, k].item()
                        if di < 0: continue
                        tc += 1
                        if torch.allclose(g_rx[tr][di].float(), g_x[sr][t].float(), atol=1e-2, rtol=1e-2):
                            tm += 1
                        elif tc - tm <= 5:
                            print(f"      MISMATCH: src={sr} t={t} k={k} -> r={tr} i={di}", flush=True)
        if tc == 0:
            print("    WARNING: no checks", flush=True)
            ok = False
        elif tm == tc:
            print(f"    PASS: recv_x all {tc} matched", flush=True)
        else:
            print(f"    FAIL: recv_x {tm}/{tc}", flush=True)
            ok = False

        if ok:
            print(f"\n  ALL PASS for {world_size}-GPU torchrun test!", flush=True)
        else:
            print(f"\n  SOME FAILED for {world_size}-GPU test!", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
