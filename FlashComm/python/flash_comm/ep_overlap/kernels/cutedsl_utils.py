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

import cutlass.cute as cute

__all__ = [
    "decode_token_src_rank_topk_and_indices",
]


@cute.jit
def decode_token_src_rank_topk_and_indices(encoded):
    """Decode ``(rank << 48) | (topk << 32) | token_idx`` metadata
    """
    buf = cute.make_rmem_tensor((4, ), cute.Int16)
    buf_i64 = cute.recast_tensor(buf, cute.Int64)
    buf_i32 = cute.recast_tensor(buf, cute.Int32)
    buf_i64[0] = encoded
    token_idx = buf_i32[0]
    topk_idx = cute.Int32(buf[2])
    rank = cute.Int32(buf[3])
    return token_idx, rank, topk_idx
