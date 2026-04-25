# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import triton
import triton.language as tl


@triton.jit
def dist_swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
):
    """communication swizzle Nz"""
    data_row_loop_num = tl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = tl.cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num
    rank_stride = rank_size // comm_npu_split
    swizzle_offset = comm_npu_split
    rank_loop_num = tl.cdiv(rank_size, swizzle_offset)
    rank_tile_idx = iter_id // (swizzle_offset * data_loop_num)
    data_rank_tile_idx = iter_id % (swizzle_offset * data_loop_num)
    rank_tile_size = swizzle_offset
    if rank_tile_idx == rank_loop_num - 1:
        rank_tile_size = rank_size - swizzle_offset * rank_tile_idx
    data_tile_idx = data_rank_tile_idx // rank_tile_size
    rank_idx = rank_tile_idx * swizzle_offset + data_rank_tile_idx % rank_tile_size
    rank_idx = (rank_idx * rank_stride) % rank_stride + (
        rank_idx * rank_stride
    ) // rank_size
    rank_idx = (rank_idx + data_tile_idx) % rank_size
    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = tl.minimum(
        data_row_shape - data_row_idx * tile_row_shape, tile_row_shape
    )
    comm_col_size = tl.minimum(
        data_col_shape - data_col_idx * tile_col_shape, tile_col_shape
    )
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    """gemm swizzle Nz"""
    data_row_loop_num = tl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = tl.cdiv(data_col_shape, tile_col_shape)
    col_loop_num = tl.cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx
