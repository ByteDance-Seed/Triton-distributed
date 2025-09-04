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

# adapted from fla
# Original Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import os
import torch
import functools

import triton
import triton.language as tl
from triton_dist.tools import aot_compile_spaces

if "USE_TRITON_DISTRIBUTED_AOT" in os.environ and os.environ["USE_TRITON_DISTRIBUTED_AOT"] in [
        "1", "true", "on", "ON", "On", True
]:
    USE_AOT = True
else:
    USE_AOT = False

if USE_AOT:
    from triton._C.libtriton_distributed import distributed


@functools.lru_cache(maxsize=4)
def prepare_lens(cu_seqlens: torch.IntTensor) -> torch.IntTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@functools.lru_cache(maxsize=4)
def prepare_chunk_indices(cu_seqlens: torch.IntTensor, chunk_size: int) -> torch.IntTensor:
    seqlens = prepare_lens(cu_seqlens)
    indices = torch.cat([torch.arange(n, dtype=torch.int32) for n in triton.cdiv(seqlens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0, dtype=torch.int32) - 1, indices], 1).to(cu_seqlens)


@functools.lru_cache(maxsize=4)
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


chunk_kkt_inv_ut_fused_kernel_signature = (
    "*{k_dtype}:16, "  # k
    "*{k_dtype}:16, "  # v
    "*{k_dtype}, "  # beta
    "*fp32, "  # g
    "*i32, "  # cu_seqlens
    "*i32, "  # chunk_indices
    "*fp32:16, "  # tmp_A
    "*fp32, "  # g_cumsum
    "*{k_dtype}:16, "  # Ai
    "*{k_dtype}:16, "  # w
    "*{k_dtype}:16, "  # u
    "i32, "  # T
    "i32, "  # NT
    "%H, "  # H
    "%K, "  # K
    "%V, "  # V
    "%BT, "  # BT
    "%BK, "  # BK
    "%BV, "  # BV
    "%USE_G"  # USE_G
)


def get_chunk_kkt_inv_ut_fused_kernel_info(H: int, K: int, V: int, USE_G: bool):
    return {
        "H": H,
        "K": K,
        "V": V,
        "BT": 64,
        "BK": 128 if K == 96 else K,
        "BV": 128 if V == 96 else V,
        "USE_G": USE_G,
        "num_warps": 4,
        "num_stages": 4,
    }


@aot_compile_spaces({
    "chunk_kkt_inv_ut_fused_kernel": {
        "signature":
        chunk_kkt_inv_ut_fused_kernel_signature.format(k_dtype="bf16"),
        "grid": ["NT", "%H", "1"],
        "triton_algo_infos": [
            get_chunk_kkt_inv_ut_fused_kernel_info(H=num_heads, K=K, V=V, USE_G=use_g)
            for (K, V) in [(96, 128), (128, 128)]
            for num_heads in [12, 16, 18, 20]
            for use_g in [True]
        ],
    }
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps, num_stages=num_stages) for num_warps in [4] for num_stages in [3]],
    key=[],
)
@triton.jit(do_not_specialize=["T"])
def chunk_kkt_inv_ut_fused_kernel(
    k,
    v,
    beta,
    g,
    cu_seqlens,
    chunk_indices,
    A,  # tmp buffer [B, T, H, BT]
    g_cumsum,  # output buffer1 [B, T, H, BT]
    Ai,  # output buffer2 [B, T, H, BT]
    w,  # output buffer3 [B, T, H, K]
    u,  # output buffer4 [B, T, H, V]
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,  # chunk size
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr = True,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H

    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # >>> g cumsum ###############################
    p_ginp = tl.make_block_ptr(g + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    p_gout = tl.make_block_ptr(g_cumsum + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    b_s = tl.load(p_ginp, boundary_check=(0, )).to(tl.float32)
    b_g = tl.cumsum(b_s, axis=0)
    tl.store(p_gout, b_g.to(p_gout.dtype.element_ty), boundary_check=(0, ))

    # >>> chunk beta g kk^t ###############################
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # b_kb = b_k * b_beta[:, None]
        b_kb = b_k
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
    b_A *= b_beta[:, None]

    if USE_G:
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * tl.exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    # >>> solve tril ###############################
    tl.static_assert(BT == 64, "assume BT == 64 for now")
    # >>> solve tril 1. diag inv ###############################
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    cur_A = A + (bos * H + i_h) * BT
    cur_Ai = Ai + (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_33 = -tl.where(m_A, tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32), 0)
    b_Ai_44 = -tl.where(m_A, tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32), 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(cur_A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(cur_A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(cur_A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(cur_A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(cur_A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision="ieee"), b_Ai_11, input_precision="ieee")
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision="ieee"), b_Ai_22, input_precision="ieee")
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision="ieee"), b_Ai_33, input_precision="ieee")

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision="ieee") + tl.dot(b_A_32, b_Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision="ieee") + tl.dot(b_A_43, b_Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision="ieee") + tl.dot(b_A_42, b_Ai_21, input_precision="ieee") +
        tl.dot(b_A_43, b_Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(cur_Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    # recompute wu ###############################
    p_A = tl.make_block_ptr(Ai + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g_exp = tl.exp(b_g)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # b_vb = b_v
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # b_kb = b_k
        b_kb = b_k * b_beta[:, None]
        if USE_G:
            b_kb = b_kb * b_g_exp[:, None]
        b_kb = b_kb.to(b_k.dtype)
        # b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def chunk_kkt_inv_ut_fused_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
):
    """
    varlen mode
    k: [1, T, H, K]
    v: [1, T, H, V]
    beta: [1, T, H]
    g: [1, T, H]
    cu_seqlens: [B + 1]
    chunk_indices: [NT * 2] where NT = number of chunks
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = 128 if K == 96 else K
    BV = 128 if V == 96 else V
    USE_G = g is not None

    assert all(x in [64, 96, 128, 256] for x in [K, V]), f"got K={K}, V={V}"

    dtype = k.dtype
    device = k.device

    g_cumsum = torch.empty([1, T, H], dtype=torch.float32, device=k.device)
    Ai = torch.zeros([1, T, H, BT], dtype=dtype, device=device)
    w = torch.empty([1, T, H, K], dtype=dtype, device=device)
    u = torch.empty([1, T, H, V], dtype=dtype, device=device)
    tmp_A = torch.empty([1, T, H, BT], dtype=torch.float32, device=device)

    NT = len(chunk_indices)
    grid = (NT, B * H)

    if USE_AOT:
        assert USE_G, "decay must be provided for AOT mode"
        assert g_cumsum.dtype == torch.float32, "`g_cumsum` in logspace must be of dtype float32"
        algo_info = distributed.chunk_kkt_inv_ut_fused_kernel__triton_algo_info_t()
        for _k, _v in get_chunk_kkt_inv_ut_fused_kernel_info(H=H, K=K, V=V, USE_G=True).items():
            setattr(algo_info, _k, _v)
        distributed.chunk_kkt_inv_ut_fused_kernel(
            0,  # torch.cuda.current_stream().cuda_stream,
            k.data_ptr(),
            v.data_ptr(),
            beta.data_ptr(),
            g.data_ptr(),
            cu_seqlens.data_ptr(),
            chunk_indices.data_ptr(),
            tmp_A.data_ptr(),
            g_cumsum.data_ptr(),
            Ai.data_ptr(),
            w.data_ptr(),
            u.data_ptr(),
            T,
            NT,
            algo_info,
        )
    else:
        chunk_kkt_inv_ut_fused_kernel[grid](
            k=k,
            v=v,
            beta=beta,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            A=tmp_A,
            g_cumsum=g_cumsum,
            Ai=Ai,
            w=w,
            u=u,
            T=T,
            NT=NT,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_G=USE_G,
        )

    return g_cumsum, Ai, w, u


chunk_gated_delta_rule_fwd_kernel_h_blockdim64_signature = (
    "*{k_dtype}:16, "  # k
    "*{k_dtype}:16, "  # v
    "*{k_dtype}:16, "  # w
    "*{k_dtype}:16, "  # v_new
    "*fp32:16, "  # g
    "*{k_dtype}:16, "  # h
    "*{k_dtype}:16, "  # h0
    "*{k_dtype}:16, "  # ht
    "*i32, "  # cu_seqlens
    "*i32, "  # chunk_offsets
    "i32, "  # T
    "i32, "  # N_MUL_H
    "%H, "  # H
    "%K, "  # K
    "%V, "  # V
    "%BT, "  # BT
    "%BV, "  # BV
    "%USE_G, "  # USE_G
    "%USE_INITIAL_STATE, "
    "%STORE_FINAL_STATE, "
    "%SAVE_NEW_VALUE")


def get_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_info(
    H: int,
    K: int,
    V: int,
    USE_G: bool,
    USE_INITIAL_STATE: bool,
    STORE_FINAL_STATE: bool,
    SAVE_NEW_VALUE: bool,
):
    return {
        "H": H,
        "K": K,
        "V": V,
        "BT": 64,
        "BV": 16,  # to increase the parallelism, we use 16 here
        "USE_G": USE_G,
        "USE_INITIAL_STATE": USE_INITIAL_STATE,
        "STORE_FINAL_STATE": STORE_FINAL_STATE,
        "SAVE_NEW_VALUE": SAVE_NEW_VALUE,
        "num_warps": 4,
        "num_stages": 4,
    }


@aot_compile_spaces({
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {
        "signature":
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64_signature.format(k_dtype="bf16"),
        "grid": ["%V / %BV", "N_MUL_H", "1"],
        "triton_algo_infos": [
            get_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_info(
                H=num_heads,
                K=K,
                V=V,
                USE_G=use_g,
                USE_INITIAL_STATE=use_initial_state,
                STORE_FINAL_STATE=store_final_state,
                SAVE_NEW_VALUE=True,
            )
            for (K, V) in [(128, 128), (96, 128)]
            for num_heads in [12, 16, 18, 20]
            for use_g in [True]
            for use_initial_state in [False, True]
            for store_final_state in [False, True]
        ],
    }
})
@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT", "USE_G"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,  # [B, T, H, K]
    v,  # [B, T, H, V]
    w,  # [B, T, H, K]
    v_new,  # [B, T, H, V]
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    N_MUL_H,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v_new = (tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV),
                                     (1, 0)) if SAVE_NEW_VALUE else None)
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
            b_g = tl.load(p_g, boundary_check=(0, ))
            b_v_new = b_v_new * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v_new)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Params:
        initial_state:
    Return:
        final_state: fp32 [N, H, K, V] if output_final_state is True, otherwise None
    """
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    if USE_AOT:
        assert g is not None
        assert save_new_value and chunk_size == 64
        algo_info = distributed.chunk_gated_delta_rule_fwd_kernel_h_blockdim64__triton_algo_info_t()
        for _k, _v in get_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_info(
                H=H,
                K=K,
                V=V,
                USE_G=g is not None,
                USE_INITIAL_STATE=initial_state is not None,
                STORE_FINAL_STATE=output_final_state,
                SAVE_NEW_VALUE=save_new_value,
        ).items():
            setattr(algo_info, _k, _v)

        distributed.chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
            0,  # torch.cuda.current_stream().cuda_stream,
            k.data_ptr(),  # k
            u.data_ptr(),  # v
            w.data_ptr(),  # w
            v_new.data_ptr(),  # v_new
            g.data_ptr(),  # g
            h.data_ptr(),  # h
            initial_state.data_ptr() if initial_state is not None else 0,  # h0
            final_state.data_ptr() if output_final_state else 0,  # ht
            cu_seqlens.data_ptr(),
            chunk_offsets.data_ptr(),
            T,
            N * H,
            algo_info,
        )
    else:
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            N_MUL_H=N * H,
            H=H,
            K=K,
            V=V,
            BT=BT,
        )
    return h, v_new, final_state


chunk_fwd_o_signature = (
    "*{k_dtype}:16, "  # q
    "*{k_dtype}:16, "  # k
    "*{k_dtype}:16, "  # v
    "*{k_dtype}:16, "  # h
    "*fp32, "  # g
    "*{k_dtype}:16, "  # o
    "*i32, "  # cu_seqlens
    "*i32, "  # chunk_indices
    "fp32, "  # scale
    "i32, "  # T
    "i32, "  # NT
    "%H, "  # H
    "%K, "  # K
    "%V, "  # V
    "%BT, "  # BT
    "%BK, "  # BK
    "%BV, "  # BV
    "%USE_G"  # USE_G
)


def get_chunk_fwd_o_info(H: int, K: int, V: int, USE_G: bool):
    return {
        "H": H,
        "K": K,
        "V": V,
        "BT": 64,
        "BK": 64,
        "BV": 64,
        "USE_G": USE_G,
        "num_warps": 4,
        "num_stages": 4,
    }


@aot_compile_spaces({
    "chunk_fwd_kernel_o": {
        "signature":
        chunk_fwd_o_signature.format(k_dtype="bf16"),
        "grid": ["%V / %BV", "NT", "%H"],
        "triton_algo_infos": [
            get_chunk_fwd_o_info(H=num_heads, K=K, V=V, USE_G=use_g)
            for (K, V) in [(128, 128), (96, 128)]
            for num_heads in [12, 16, 18, 20]
            for use_g in [True]
        ],
    }
})
@triton.heuristics({"USE_G": lambda args: args["g"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for BV in [64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    # NT = tl.cdiv(T, BT)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T, ), (H, ), (i_t * BT, ), (BT, ), (0, ))
        b_g = tl.load(p_g, boundary_check=(0, ))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    # o_i = tl.arange(0, BT)
    # m_A = o_i[:, None] >= o_i[None, :]
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1]**-0.5

    o = torch.empty_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    if USE_AOT:
        assert cu_seqlens is not None, "only varlen is supported for AOT mode"
        algo_info = distributed.chunk_fwd_kernel_o__triton_algo_info_t()
        for _k, _v in get_chunk_fwd_o_info(H, K, V, g is not None).items():
            setattr(algo_info, _k, _v)
        distributed.chunk_fwd_kernel_o(
            torch.cuda.current_stream().cuda_stream,
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            h.data_ptr(),
            g.data_ptr(),
            o.data_ptr(),
            cu_seqlens.data_ptr(),
            chunk_indices.data_ptr(),
            scale,
            T,
            NT,
            algo_info,
        )
    else:
        chunk_fwd_kernel_o[grid](
            q,
            k,
            v,
            h,
            g,
            o,
            cu_seqlens,
            chunk_indices,
            scale,
            T=T,
            NT=NT,
            H=H,
            K=K,
            V=V,
            BT=BT,
        )
    return o


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.IntTensor] = None,
    return_chunked_states: bool = False,
):
    assert cu_seqlens is not None, "assume varlen mode for now"
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    g, A, w, u = chunk_kkt_inv_ut_fused_triton(
        k=k,
        v=v,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if return_chunked_states:
        return g, o, A, final_state, h
    return g, o, A, final_state


if __name__ == "__main__":
    import torch.nn.functional as F
    from triton_dist.utils import perf_func
    torch.manual_seed(0)
    torch.set_default_device("cuda")

    dtype = torch.bfloat16

    CHUNK_SIZE = 64
    B, T, H, K, V = 1, 8192, 16, 128, 128
    k = F.normalize(torch.randn(1, T, H, K, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn([B, T, H, V], dtype=dtype)
    beta = torch.randn([B, T, H], dtype=dtype).sigmoid()
    g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))

    cu_seqlens = torch.tensor([0, T], dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)

    # benchmark


    def unfused_fla(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cu_seqlens: torch.Tensor,
        chunk_indices: torch.Tensor,
    ):
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
        from fla.ops.utils import chunk_local_cumsum, solve_tril

        g_cumsum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g_cumsum,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        Ai = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=Ai,
            g_cumsum=g_cumsum,
            cu_seqlens=cu_seqlens,
        )
        return g_cumsum, Ai, w, u, A

    def check_res(res: tuple[torch.Tensor], ref: tuple[torch.Tensor], s: str):
        var_name = ["g_cumsum", "Ai", "w", "u", "A"]
        for r, rf, n in zip(res, ref, var_name):
            try:
                r_norm = r.abs().mean()
                rf_norm = rf.abs().mean()
                diff_acc = r.sub(rf).abs().sum()

                if n == "Ai" and T % 64 == 0:
                    identity = torch.eye(64, dtype=torch.float32)
                    a_src = x_fla[-1] if ("fused" in s) else res[-1]
                    A_res_reshape = a_src[-1].reshape([B, T // 64, 64, H, 64]).transpose(2, 3) + torch.eye(
                        64, dtype=torch.float32)
                    Ai_res_reshape = res[1].reshape([B, T // 64, 64, H, 64]).transpose(2, 3).float()

                    A_ref_reshape = ref[-1].reshape([B, T // 64, 64, H, 64]).transpose(2, 3) + torch.eye(
                        64, dtype=torch.float32)
                    Ai_ref_reshape = ref[1].reshape([B, T // 64, 64, H, 64]).transpose(2, 3).float()

                    res_inv_diff_acc = ((A_res_reshape @ Ai_res_reshape) - identity).abs().sum()
                    ref_inv_diff_acc = ((A_ref_reshape @ Ai_ref_reshape) - identity).abs().sum()

                    print(f">>> res_inv_diff_acc: {res_inv_diff_acc}, ref_inv_diff_acc: {ref_inv_diff_acc}")

                torch.testing.assert_close(r, rf, atol=1e-4, rtol=1e-4)
                print(f"✅ `{s}` {n} match: {r.shape} ({r_norm}) vs {rf.shape} ({rf_norm}), diff_acc: {diff_acc}")
            except AssertionError as e:
                print(f"❌ `{s}` {n} mismatch: {r.shape} ({r_norm}) vs {rf.shape} ({rf_norm}), diff_acc: {diff_acc}")
                print(e)
        print("\n")

    warmup_iters = 20
    bench_iters = 200

    # 0. unfused fla
    unfused_fla = functools.partial(unfused_fla, k=k, v=v, beta=beta, g=g, cu_seqlens=cu_seqlens,
                                    chunk_indices=chunk_indices)
    x_fla, t_fla = perf_func(unfused_fla, warmup_iters=warmup_iters, iters=bench_iters)
    print(f"{'gdn_unfused_fla':<20} {T:<8}: {t_fla:.3f} ms")

    # 1. fused Triton
    fused_triton = functools.partial(chunk_kkt_inv_ut_fused_triton, k=k, v=v, beta=beta, g=g, cu_seqlens=cu_seqlens,
                                     chunk_indices=chunk_indices)
    x_fused_triton, t_ftriton = perf_func(fused_triton, warmup_iters=warmup_iters, iters=bench_iters)
    print(f"{'gdn_fused_triton':<20} {T:<8}: {t_ftriton:.3f} ms")

    ref = x_fla
    for func_name, res in [("fused_triton", x_fused_triton)]:
        check_res(res, ref, func_name)
