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
import triton
import triton.language as tl
from .task_context import TaskBaseInfo, Scoreboard, TensorDesc
from .utils import next_power_of_2


@triton.jit
def rmsnorm_rope_update_kv_cache_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,
    Q_RMS_EPS: tl.constexpr,
    K_RMS_EPS: tl.constexpr,
):
    # cos/sin: [sin_cos_batch/1, seq_len, q_head_dim]
    # qkv : [batch, seq_len, num_total_heads, q_head_dim]
    # rms_weight: [q_head_dim]
    # assume that q_head_dim == v_head_dim

    qkv_tensor: TensorDesc = task_base_info.get_tensor(0)
    block_tables_tensor: TensorDesc = task_base_info.get_tensor(1)
    kv_lens_tensor: TensorDesc = task_base_info.get_tensor(2)
    q_rms_weight_tensor: TensorDesc = task_base_info.get_tensor(3)
    k_rms_weight_tensor: TensorDesc = task_base_info.get_tensor(4)
    cos_cache_tensor: TensorDesc = task_base_info.get_tensor(5)
    sin_cache_tensor: TensorDesc = task_base_info.get_tensor(6)
    k_cache_tensor: TensorDesc = task_base_info.get_tensor(7)
    v_cache_tensor: TensorDesc = task_base_info.get_tensor(8)
    q_norm_rope_tensor: TensorDesc = task_base_info.get_tensor(9)

    # num tiles of qkv
    tile_id = task_base_info.tile_id_or_start

    k_cache_ptr = k_cache_tensor.data_ptr(tl.bfloat16)
    v_cache_ptr = v_cache_tensor.data_ptr(tl.bfloat16)

    q_rms_weight_ptr = q_rms_weight_tensor.data_ptr(tl.bfloat16)
    k_rms_weight_ptr = k_rms_weight_tensor.data_ptr(tl.bfloat16)

    sin_cos_batch = cos_cache_tensor.size(0)
    seq_len = qkv_tensor.size(1)
    qkv_ptr = qkv_tensor.data_ptr(tl.bfloat16)
    q_norm_rope_ptr = q_norm_rope_tensor.data_ptr(tl.bfloat16)
    sin_ptr = sin_cache_tensor.data_ptr(tl.float32)
    cos_ptr = cos_cache_tensor.data_ptr(tl.float32)
    kv_lens_ptr = kv_lens_tensor.data_ptr(tl.int32)
    block_table_ptr = block_tables_tensor.data_ptr(tl.int32)

    tl.static_assert(Q_HEAD_DIM == V_HEAD_DIM)
    num_total_heads: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS * 2
    num_qk_heads: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS

    q_head_dim: tl.constexpr = Q_HEAD_DIM
    PADDED_Q_HEAD_DIM: tl.constexpr = next_power_of_2(Q_HEAD_DIM)
    PADDED_V_NUM_HEADS: tl.constexpr = next_power_of_2(NUM_KV_HEADS)
    K_HEAD_DIM: tl.constexpr = Q_HEAD_DIM

    cols = tl.arange(0, PADDED_Q_HEAD_DIM)
    offs_vh = tl.arange(0, PADDED_V_NUM_HEADS)

    stride_table_bs: tl.constexpr = MAX_NUM_BLOCKS_PER_SEQ
    stride_value_per_token: tl.constexpr = V_HEAD_DIM * NUM_KV_HEADS
    stride_qkv_batch = seq_len * num_total_heads * V_HEAD_DIM
    stride_qkv_token: tl.constexpr = V_HEAD_DIM * num_total_heads
    stride_key_per_token: tl.constexpr = K_HEAD_DIM * NUM_KV_HEADS

    # each tile is responsible for a head
    idx_0 = tile_id // num_qk_heads
    idx_1 = tile_id % num_qk_heads
    batch_idx = idx_0 // seq_len
    cos_row_idx = idx_0 % seq_len
    seq_idx = cos_row_idx
    # kv len has been added with the current seq_len
    history_kv = tl.load(kv_lens_ptr + batch_idx) - seq_len
    # set value_cache in tile which calculate key
    if idx_1 >= NUM_Q_HEADS:
        global_seq_id = seq_idx + history_kv
        page_id = global_seq_id // PAGE_SIZE
        value_offs = offs_vh[:, None] * V_HEAD_DIM + cols[None:]
        value_mask = (offs_vh[:, None] < NUM_KV_HEADS) & (cols[None:] < V_HEAD_DIM)
        block_entry_id = tl.load(block_table_ptr + batch_idx * stride_table_bs + page_id)
        value_tile = tl.load(
            qkv_ptr + batch_idx * stride_qkv_batch + seq_idx * stride_qkv_token + num_qk_heads * Q_HEAD_DIM +
            value_offs, mask=value_mask, other=0)
        tl.store(v_cache_ptr + (block_entry_id + global_seq_id % PAGE_SIZE) * stride_value_per_token + value_offs,
                 value_tile, mask=value_mask)

    X = qkv_ptr + idx_0 * num_total_heads * q_head_dim + idx_1 * q_head_dim
    q_norm_rope_out_ptr = q_norm_rope_ptr + idx_0 * NUM_Q_HEADS * q_head_dim + idx_1 * q_head_dim

    if idx_1 >= NUM_Q_HEADS:
        RMS_EPS = K_RMS_EPS
        rms_weight_ptr = k_rms_weight_ptr
    else:
        RMS_EPS = Q_RMS_EPS
        rms_weight_ptr = q_rms_weight_ptr
    # apply rmsnorm in qk head_dim
    _rms = tl.zeros([PADDED_Q_HEAD_DIM], dtype=tl.float32)

    a = tl.load(X + cols, mask=cols < q_head_dim, other=0.0).to(tl.float32)
    _rms += a * a
    rms = tl.rsqrt(tl.sum(_rms) / q_head_dim + RMS_EPS)

    mask = cols < q_head_dim
    w = tl.load(rms_weight_ptr + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    y = w * (x * rms)

    # store rmsnorm for debug
    # tl.store(Y + cols, y, mask=mask)
    # apply rope in qk head_dim
    # if sin_cos_batch == 1, each request share same sin/cos
    if sin_cos_batch == 1:
        cos_ptr = cos_ptr + (cos_row_idx + history_kv) * q_head_dim
        sin_ptr = sin_ptr + (cos_row_idx + history_kv) * q_head_dim
    else:
        cos_ptr = cos_ptr + batch_idx * (seq_len * q_head_dim) + (cos_row_idx + history_kv) * q_head_dim
        sin_ptr = sin_ptr + batch_idx * (seq_len * q_head_dim) + (cos_row_idx + history_kv) * q_head_dim

    cos_offsets = tl.arange(0, PADDED_Q_HEAD_DIM // 2)
    cos_mask = cos_offsets < q_head_dim // 2
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0)

    y = tl.reshape(y, (2, PADDED_Q_HEAD_DIM // 2))
    y = tl.permute(y, (1, 0))
    y0, y1 = tl.split(y)
    first_half_qk_offsets = tl.arange(0, PADDED_Q_HEAD_DIM // 2)
    first_qk_mask = tl.arange(0, PADDED_Q_HEAD_DIM // 2) < q_head_dim // 2
    # qkv_tile_1 = tl.load(X + first_half_qk_offsets, mask=first_qk_mask, other=0).to(sin_row.dtype)

    # # right half of the head
    second_half_qk_offsets = first_half_qk_offsets + (q_head_dim // 2)
    second_qk_mask = first_qk_mask
    # qkv_tile_2 = tl.load(X + second_half_qk_offsets, mask=second_qk_mask, other=0).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # cast to bf16 to keep same behavior
    y0 = y0.to(tl.bfloat16)
    y1 = y1.to(tl.bfloat16)
    new_qkv_tile_0 = y0 * cos_row - y1 * sin_row
    new_qkv_tile_1 = y1 * cos_row + y0 * sin_row
    if idx_1 < NUM_Q_HEADS:
        tl.store(q_norm_rope_out_ptr + first_half_qk_offsets, new_qkv_tile_0, mask=first_qk_mask)
        tl.store(q_norm_rope_out_ptr + second_half_qk_offsets, new_qkv_tile_1, mask=second_qk_mask)

    # set key cache
    if idx_1 >= NUM_Q_HEADS:
        global_seq_id = seq_idx + history_kv
        page_id = global_seq_id // PAGE_SIZE
        key_head_idx = idx_1 - NUM_Q_HEADS
        block_entry_id = tl.load(block_table_ptr + batch_idx * stride_table_bs + page_id)
        tl.store(
            k_cache_ptr + (block_entry_id + global_seq_id % PAGE_SIZE) * stride_key_per_token +
            key_head_idx * K_HEAD_DIM + first_half_qk_offsets, new_qkv_tile_0, mask=first_qk_mask)
        tl.store(
            k_cache_ptr + (block_entry_id + global_seq_id % PAGE_SIZE) * stride_key_per_token +
            key_head_idx * K_HEAD_DIM + second_half_qk_offsets, new_qkv_tile_1, mask=second_qk_mask)

    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def rmsnorm_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    RMS_EPS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tile_id = task_base_info.tile_id_or_start
    row = tile_id
    input_tensor: TensorDesc = task_base_info.get_tensor(0)
    weight_tensor: TensorDesc = task_base_info.get_tensor(1)
    output_tensor: TensorDesc = task_base_info.get_tensor(2)
    input_ptr = input_tensor.data_ptr(tl.bfloat16)
    weight_ptr = weight_tensor.data_ptr(tl.bfloat16)
    output_ptr = output_tensor.data_ptr(tl.bfloat16)
    N = output_tensor.size(1, 16)

    Y = output_ptr + row * N
    X = input_ptr + row * N

    square_sum = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        square_sum += x * x
    rms = tl.rsqrt(tl.sum(square_sum) / N + RMS_EPS)

    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        w = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = w * (x * rms)
        tl.store(Y + cols, y, mask=mask)

    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def _qkv_pack_qk_norm_rope_split_v_kernel(tile_id, qkv_ptr, kv_lens_ptr, q_rms_weight_ptr, k_rms_weight_ptr,  #
                                          q_rms_eps, k_rms_eps, sin_ptr, cos_ptr,  #
                                          q_norm_rope_ptr, k_norm_rope_ptr, v_ptr,  #
                                          seq_len, num_q_heads, num_kv_heads,  #
                                          head_dim, BLOCK_SEQ: tl.constexpr, BLOCK_HD: tl.constexpr):
    """
        qkv: (bs, seq_len, num_total_heads, head_dim)
        BLOCK_HD equal to next_power_of_2(head_dim)
    """
    pid = tile_id
    num_total_heads = num_q_heads + 2 * num_kv_heads
    group_size = num_q_heads // num_kv_heads
    head_type = pid % (group_size + 2)
    num_tiles_seq = tl.cdiv(seq_len, BLOCK_SEQ)
    tile_id_bs_seq = pid // num_total_heads
    tile_id_bs = tile_id_bs_seq // num_tiles_seq
    tild_id_seq = tile_id_bs_seq % num_tiles_seq

    head_group_id = (pid % num_total_heads) // (group_size + 2)
    offs_seq = tl.arange(0, BLOCK_SEQ) + tild_id_seq * BLOCK_SEQ
    offs_hd = tl.arange(0, BLOCK_HD)
    stride_qkv_seq = num_total_heads * head_dim
    stride_nh = head_dim
    bs_idx = tile_id_bs.to(tl.int64)
    history_kv = tl.load(kv_lens_ptr + bs_idx)
    mask = (offs_seq[:, None] < seq_len) & (offs_hd[None, :] < head_dim)

    # split v
    if head_type == group_size + 1:
        head_id_no_pack = head_group_id
        head_id_pack = head_id_no_pack + num_q_heads + num_kv_heads
        v_input_ptrs = qkv_ptr + bs_idx * seq_len * stride_qkv_seq + offs_seq[:,
                                                                              None] * stride_qkv_seq + head_id_pack * stride_nh + offs_hd[
                                                                                  None:]
        v_val = tl.load(v_input_ptrs, mask=mask)
        v_out_ptrs = v_ptr + bs_idx * seq_len * num_kv_heads * head_dim + offs_seq[:,
                                                                                   None] * num_kv_heads * head_dim + head_id_no_pack * stride_nh + offs_hd[
                                                                                       None:]
        tl.store(v_out_ptrs, v_val, mask=mask)
    else:
        # q/k norm & rope
        if head_type < group_size:
            head_id_no_pack = head_group_id * group_size + head_type
            head_id_pack = head_id_no_pack
            RMS_EPS = q_rms_eps
            rms_weight_ptr = q_rms_weight_ptr
            stride_seq = num_q_heads * head_dim
            out_ptr = q_norm_rope_ptr + bs_idx * seq_len * stride_seq + head_id_no_pack * head_dim
        else:
            head_id_no_pack = head_group_id
            head_id_pack = head_id_no_pack + num_q_heads
            RMS_EPS = k_rms_eps
            rms_weight_ptr = k_rms_weight_ptr
            stride_seq = num_kv_heads * head_dim
            out_ptr = k_norm_rope_ptr + bs_idx * seq_len * stride_seq + head_id_no_pack * head_dim

        # apply rmsnorm in qk head_dim
        _rms = tl.zeros([BLOCK_SEQ, BLOCK_HD], dtype=tl.float32)
        input_ptrs = qkv_ptr + bs_idx * seq_len * stride_qkv_seq + offs_seq[:,
                                                                            None] * stride_qkv_seq + head_id_pack * stride_nh + offs_hd[
                                                                                None:]
        a = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        _rms += a * a  # [BLOCK_SEQ, BLOCK_HD]
        rms = tl.rsqrt(tl.sum(_rms, axis=1, keep_dims=True) / head_dim + RMS_EPS)  # [BLOCK_SEQ, 1]

        mask_hd = (offs_hd < head_dim)
        w = tl.load(rms_weight_ptr + offs_hd, mask=mask_hd).to(tl.float32)  # [BLOCK_HD]
        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_SEQ, BLOCK_HD]
        y = w[None, :] * (x * rms)  # [BLOCK_SEQ, BLOCK_HD]

        cos_offsets = tl.arange(0, BLOCK_HD // 2)
        cos_ptrs = cos_ptr + (offs_seq[:, None] + history_kv) * head_dim + cos_offsets[None, :]
        sin_ptrs = sin_ptr + (offs_seq[:, None] + history_kv) * head_dim + cos_offsets[None, :]

        cos_mask = (offs_seq[:, None] < seq_len) & (cos_offsets[None, :] < head_dim // 2)
        cos_row = tl.load(cos_ptrs, mask=cos_mask, other=0)  # [BLOCK_SEQ, BLOCK_HD // 2]
        sin_row = tl.load(sin_ptrs, mask=cos_mask, other=0)  # [BLOCK_SEQ, BLOCK_HD // 2]

        y = tl.reshape(y, (BLOCK_SEQ, 2, BLOCK_HD // 2))
        y = tl.permute(y, (0, 2, 1))
        y0, y1 = tl.split(y)  # [BLOCK_SEQ, BLOCK_HD // 2]
        first_half_qk_offsets = offs_seq[:, None] * stride_seq + cos_offsets
        first_qk_mask = cos_mask

        # # right half of the head
        second_half_qk_offsets = first_half_qk_offsets + (head_dim // 2)
        second_qk_mask = first_qk_mask
        # qkv_tile_2 = tl.load(X + second_half_qk_offsets, mask=second_qk_mask, other=0).to(sin_row.dtype)

        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # cast to bf16 to keep same behavior
        y0 = y0.to(tl.bfloat16)
        y1 = y1.to(tl.bfloat16)
        new_qkv_tile_0 = y0 * cos_row - y1 * sin_row
        new_qkv_tile_1 = y1 * cos_row + y0 * sin_row

        tl.store(out_ptr + first_half_qk_offsets, new_qkv_tile_0, mask=first_qk_mask)
        tl.store(out_ptr + second_half_qk_offsets, new_qkv_tile_1, mask=second_qk_mask)


@triton.jit
def qkv_pack_qk_norm_rope_split_v_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    DTYPE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_RMS_EPS: tl.constexpr,
    K_RMS_EPS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):

    tile_id = task_base_info.tile_id_or_start
    qkv_tensor: TensorDesc = task_base_info.get_tensor(0)
    kv_lens_tensor: TensorDesc = task_base_info.get_tensor(1)
    q_rms_weight_tensor: TensorDesc = task_base_info.get_tensor(2)
    k_rms_weight_tensor: TensorDesc = task_base_info.get_tensor(3)
    cos_tensor: TensorDesc = task_base_info.get_tensor(4)
    sin_tensor: TensorDesc = task_base_info.get_tensor(5)
    q_norm_rope_tensor: TensorDesc = task_base_info.get_tensor(6)
    k_norm_rope_tensor: TensorDesc = task_base_info.get_tensor(7)
    v_tensor: TensorDesc = task_base_info.get_tensor(8)
    qkv_ptr = qkv_tensor.data_ptr(DTYPE)
    kv_lens_ptr = kv_lens_tensor.data_ptr(tl.int32)
    q_rms_weight_ptr = q_rms_weight_tensor.data_ptr(DTYPE)
    k_rms_weight_ptr = k_rms_weight_tensor.data_ptr(DTYPE)
    sin_ptr = sin_tensor.data_ptr(tl.float32)
    cos_ptr = cos_tensor.data_ptr(tl.float32)
    q_norm_rope_ptr = q_norm_rope_tensor.data_ptr(DTYPE)
    k_norm_rope_ptr = k_norm_rope_tensor.data_ptr(DTYPE)
    v_ptr = v_tensor.data_ptr(DTYPE)

    # [bs, seq_len, num_total_heads, head_dim]
    seq_len = qkv_tensor.size(1)

    _qkv_pack_qk_norm_rope_split_v_kernel(tile_id, qkv_ptr, kv_lens_ptr, q_rms_weight_ptr, k_rms_weight_ptr,  #
                                          Q_RMS_EPS, K_RMS_EPS, sin_ptr, cos_ptr,  #
                                          q_norm_rope_ptr, k_norm_rope_ptr, v_ptr,  #
                                          seq_len, NUM_Q_HEADS, NUM_KV_HEADS,  #
                                          HEAD_DIM,  #
                                          BLOCK_SEQ=BLOCK_SEQ,  #
                                          BLOCK_HD=BLOCK_HD)

    scoreboard.release_tile(task_base_info, tile_id)
