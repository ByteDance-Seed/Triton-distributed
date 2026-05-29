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

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class MoETopkReduceBlockPerToken:
    """Non-persistent topk reduce: one CTA per output token.
    """

    num_threads: int = 256
    atom_v: int = 8

    @cute.jit
    def __call__(
        self,
        staging: cute.Tensor,  # (num_tokens * topk, hidden_size)
        topk_indices: cute.Tensor,  # (num_tokens, topk)
        output: cute.Tensor,  # (num_tokens, hidden_size)
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        num_experts: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        tile_n = self.num_threads * self.atom_v
        assert hidden_size % tile_n == 0, (f"hidden_size={hidden_size} must be a multiple of "
                                           f"num_threads*atom_v={tile_n}")
        num_tiles = hidden_size // tile_n
        self.kernel(
            staging,
            topk_indices,
            output,
            hidden_size,
            topk,
            num_experts,
            num_tokens,
            tile_n,
            num_tiles,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        staging: cute.Tensor,
        topk_indices: cute.Tensor,
        output: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        num_experts: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
        tile_n: cutlass.Constexpr[int],
        num_tiles: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        token_idx = bidx

        atom_v: cutlass.Constexpr[int] = self.atom_v
        num_threads: cutlass.Constexpr[int] = self.num_threads

        # 128 b atoms map to one LDG.E.128 / STG.E.128 per thread.
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            staging.element_type,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            output.element_type,
            num_bits_per_copy=128,
        )

        # Cache topk indices in registers; all threads in the CTA share
        # one L1 line so the HBM cost is negligible.
        my_topk = cute.make_fragment(topk, cutlass.Int32)
        for k in cutlass.range_constexpr(topk):
            my_topk[k] = topk_indices[token_idx, k]

        # zipped_divide by (1, atom_v): mode-0 = atom shape (1, atom_v),
        # mode-1 = (num_rows, atoms_per_row). Slicing (0, None) collapses
        # the M-atom mode to a 1D (atom_v,) view consumed directly by
        # cute.copy.
        g_out = cute.zipped_divide(output, (1, atom_v))
        g_stage = cute.zipped_divide(staging, (1, atom_v))

        atom_frag_shape = (atom_v, )
        tCrC = cute.make_fragment(atom_frag_shape, output.element_type)
        tCrAcc = cute.make_fragment(atom_frag_shape, cutlass.Float32)

        tCrR_per_k = [cute.make_fragment(atom_frag_shape, staging.element_type) for _ in range(topk)]

        for tile_idx in cutlass.range_constexpr(num_tiles):
            my_atom_idx = tile_idx * num_threads + tidx

            for k in cutlass.range_constexpr(topk):
                expert_idx = my_topk[k]
                if expert_idx < num_experts:
                    stage_row_idx = token_idx * topk + k
                    cute.copy(
                        load_atom,
                        g_stage[(0, None), (stage_row_idx, my_atom_idx)],
                        tCrR_per_k[k],
                    )

            tCrAcc.fill(0.0)
            for k in cutlass.range_constexpr(topk):
                expert_idx = my_topk[k]
                if expert_idx < num_experts:
                    tCrAcc.store(tCrAcc.load() + tCrR_per_k[k].load().to(cutlass.Float32))

            tCrC.store(tCrAcc.load().to(output.element_type))
            cute.copy(
                store_atom,
                tCrC,
                g_out[(0, None), (token_idx, my_atom_idx)],
            )
