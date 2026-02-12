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

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blockM", type=int, default=128)
    parser.add_argument("--blockN", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--swizzle", type=int, default=128)
    return parser.parse_args()


args = get_args()

BLOCK_M = args.blockM
BLOCK_N = args.blockN
SWIZZLE = args.swizzle
T = getattr(ll, args.dtype)


@lk.ll_kernel(backend="cuda", is_entry=True)
def bench_tma_2d_layout_kernel(
    input_desc: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    M: ll.int32,
    N: ll.int32,
    output: ll.Tensor[T],
) -> ll.void:
    BLOCK_ATOM = SWIZZLE // ll.sizeof(T) if SWIZZLE != 0 else BLOCK_N
    assert BLOCK_N % BLOCK_ATOM == 0, f"BLOCK_N {BLOCK_N} must be divisible by BLOCK_ATOM {BLOCK_ATOM}"

    bid = ll.blockIdx_x()
    num_blocks_n = ll.cdiv(N, BLOCK_N)
    bid_m = bid // num_blocks_n
    bid_n = bid % num_blocks_n
    warp_idx = ll.__shfl_sync("0xffffffff", ll.threadIdx_x() // 32, 0)
    barrier = ll.empty([1], dtype=ll.uint64, scope="shared")
    smem_data = ll.empty([BLOCK_M * BLOCK_N], dtype=T, scope="shared")
    ll.align_memory(1024, scope="shared")
    if warp_idx == 0 and ll.elect_one_sync():
        ll.prefetch_tma_descriptor(input_desc)
        ll.init_smem_barrier(barrier, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()

    phase = 0

    if warp_idx == 1 and ll.elect_one_sync():
        for i in ll.unroll(range(BLOCK_N // BLOCK_ATOM)):
            ll.tma_load_2d(input_desc, barrier, smem_data + i * BLOCK_ATOM * BLOCK_M, bid_n * BLOCK_N + i * BLOCK_ATOM,
                           bid_m * BLOCK_M)
        ll.mbarrier_arrive_and_expect_tx(barrier, BLOCK_M * BLOCK_N * ll.sizeof(T))

    ll.mbarrier_wait(barrier, phase)
    tid = ll.threadIdx_x()
    for i in range(tid, BLOCK_M * BLOCK_N, ll.blockDim_x()):
        data = smem_data[i]
        m = bid_m * BLOCK_M + i % (BLOCK_M * BLOCK_ATOM) // BLOCK_ATOM
        n = bid_n * BLOCK_N + i // (BLOCK_M * BLOCK_ATOM) * BLOCK_ATOM + i % (BLOCK_M * BLOCK_ATOM) % BLOCK_ATOM
        output[m * N + n] = data


def visualize_swizzle_pattern(original: torch.Tensor, swizzled: torch.Tensor, block_m: int, block_n: int,
                              swizzle_bytes: int, elem_size: int, save_path: str = "swizzle_pattern.png"):
    """
    Visualize the swizzle pattern: each element has a fixed color and number,
    showing before/after swizzle positions side by side.
    """
    # Use first 16x16 for visualization
    num_rows = 16
    num_cols = 16

    orig_tile = original[:num_rows, :num_cols].cpu().numpy().astype(np.int32)
    swiz_tile = swizzled[:num_rows, :num_cols].cpu().numpy().astype(np.int32)

    # Create distinct colors for each element value (0-15)
    # Using a perceptually distinct colormap
    cmap = plt.cm.get_cmap('tab20')
    element_colors = {i: cmap(i / 16) for i in range(16)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f'TMA Swizzle Pattern: {swizzle_bytes}B Swizzle\n'
        f'Each element has fixed color, number shows original column index', fontsize=14, fontweight='bold')

    # Plot original layout (before swizzle)
    for row in range(num_rows):
        for col in range(num_cols):
            val = orig_tile[row, col]
            color = element_colors[val % 16]
            rect = Rectangle((col, num_rows - 1 - row), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            # Text color: white for dark backgrounds, black for light
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
            ax1.text(col + 0.5, num_rows - 0.5 - row, str(val), ha='center', va='center', fontsize=9, fontweight='bold',
                     color=text_color)

    ax1.set_xlim(0, num_cols)
    ax1.set_ylim(0, num_rows)
    ax1.set_aspect('equal')
    ax1.set_title('Original Layout (GMEM)\nAll rows identical: 0,1,2,...,15', fontsize=12)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_xticks(np.arange(0.5, num_cols, 1))
    ax1.set_xticklabels(range(num_cols))
    ax1.set_yticks(np.arange(0.5, num_rows, 1))
    ax1.set_yticklabels(range(num_rows - 1, -1, -1))

    # Plot swizzled layout (after swizzle)
    for row in range(num_rows):
        for col in range(num_cols):
            val = swiz_tile[row, col]
            color = element_colors[val % 16]
            rect = Rectangle((col, num_rows - 1 - row), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            ax2.add_patch(rect)
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
            ax2.text(col + 0.5, num_rows - 0.5 - row, str(val), ha='center', va='center', fontsize=9, fontweight='bold',
                     color=text_color)

    ax2.set_xlim(0, num_cols)
    ax2.set_ylim(0, num_rows)
    ax2.set_aspect('equal')
    ax2.set_title('Swizzled Layout (SMEM)\nElements permuted per row via XOR', fontsize=12)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_xticks(np.arange(0.5, num_cols, 1))
    ax2.set_xticklabels(range(num_cols))
    ax2.set_yticks(np.arange(0.5, num_rows, 1))
    ax2.set_yticklabels(range(num_rows - 1, -1, -1))

    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=element_colors[i], edgecolor='black', label=f'{i}') for i in range(16)
    ]
    fig.legend(handles=legend_elements, loc='center right', title='Element\nValue', ncol=1, fontsize=8,
               bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Swizzle visualization saved to: {save_path}")
    plt.close()

    return save_path


if __name__ == "__main__":
    passes = PASSES["cuda"]
    code = bench_tma_2d_layout_kernel.compile(passes, codegen_cuda)
    print(code)

    M = 1024
    N = 1024
    dtype = ll.ll_type_to_torch_type(T)
    a = torch.arange(N, dtype=torch.int32, device="cuda").repeat(M).reshape(M, N).to(dtype)
    out = torch.empty(M, N, dtype=dtype, device="cuda")
    tensor_map_a = create_tma_2d_descriptor(
        a,
        gmem_inner_dim=N,
        gmem_outer_dim=M,
        smem_inner_dim=BLOCK_N,
        smem_outer_dim=BLOCK_M,
        gmem_outer_stride=N,  # stride in elements
        swizzle_mode=SWIZZLE,
    )
    grid = (ll.cdiv(M, BLOCK_M) * ll.cdiv(N, BLOCK_N), 1, 1)
    block = (128, 1, 1)
    smem_size = ll.align_power_of_2(BLOCK_M * BLOCK_N * ll.sizeof(T), 1024)
    kernel = bench_tma_2d_layout_kernel.build(passes, codegen_cuda, grid=grid, block=block, shared_mem_bytes=smem_size,
                                              arch="sm_90a", verbose=True)
    kernel(tensor_map_a, M, N, out)

    torch.set_printoptions(precision=4, threshold=float('inf'), linewidth=120, sci_mode=False)
    vec_len = 16 // dtype.itemsize
    a_view = a.view(M, -1, vec_len)[:, :, 0].view(M, -1) / vec_len
    out_view = out.view(M, -1, vec_len)[:, :, 0].view(M, -1) / vec_len
    print("Original layout (first 16x16 elements, grouped by 128-bit):")
    print(a_view[:16, :16].to(torch.int32))
    print("\nSwizzled layout (first 16x16 elements, grouped by 128-bit):")
    print(out_view[:16, :16].to(torch.int32))

    # Generate visualization
    elem_size = dtype.itemsize
    visualize_swizzle_pattern(a_view.to(torch.float32), out_view.to(torch.float32), block_m=BLOCK_M,
                              block_n=BLOCK_N // (16 // elem_size),  # Adjust for 128-bit grouping
                              swizzle_bytes=SWIZZLE, elem_size=16,  # Using 128-bit groups
                              save_path=f"swizzle_pattern_{BLOCK_M}x{BLOCK_N}_swizzle{SWIZZLE}B_{args.dtype}.png")
