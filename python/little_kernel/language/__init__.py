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

# ruff: noqa: F401, F403
################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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

from little_kernel.core.type_system import *
from .dsl import *
from ..atom.scheduler import Scheduler, GemmType, IndexType

# Export Tuple and template for type annotations
from little_kernel.core.type_system import Tuple, template

# Arithmetic
from .intrin.arith import (cdiv, align_power_of_2, min_val, max_val, ldg, bf16_to_float, float_to_bf16)

# Data type utilities
from .intrin.dtype import (sizeof, typeof, val_cast, ptr_cast, to, ll_type_to_torch_type)

# Shuffle
from .intrin.shuffle import (
    __shfl_sync,
    shfl_xor_sync,
    shfl_up_sync,
    shfl_down_sync,
)

# SIMT: thread/block indices, warp ops, register management
from .intrin.simt import (
    threadIdx_x,
    threadIdx_y,
    threadIdx_z,
    blockIdx_x,
    blockIdx_y,
    blockIdx_z,
    blockDim_x,
    blockDim_y,
    blockDim_z,
    gridDim_x,
    gridDim_y,
    gridDim_z,
    thread_y,
    thread_z,
    get_lane_idx,
    smid,
    elect_one_sync,
    ballot_sync,
    match_any_sync,
    popc,
    ffs,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)

# Memory: allocation, pointer casting, STSM
from .intrin.memory import (
    __ldcg,
    __ldca,
    empty,
    align_memory,
    alloc_dynamic_shared_memory,
    alloc_local_memory,
    alloc_shared_memory,
    slice_dynamic_shared_memory,
    smem_bf16_ptr,
    smem_u64_ptr,
    ptr_byte_offset,
    cvta_generic_to_shared,
    stsm_x2_from_floats,
)

# Barrier and synchronization
from .intrin.barrier import (
    __syncwarp,
    block_sync,
    __syncthreads,
    cluster_arrive,
    cluster_wait,
    cluster_sync,
    cluster_rank,
    init_smem_barrier,
    fence_smem_barrier_init,
    mbarrier_arrive,
    mbarrier_arrive_and_expect_tx,
    mbarrier_wait,
    mbarrier_arrive_remote,
    named_barrier_sync,
    named_barrier_arrive,
    threadfence,
    threadfence_system,
    fence_proxy_async,
    fence_async_shared,
    grid_sync,
)

# Atomic operations
from .intrin.atomic import (
    atomic_add,
    atomic_cas,
    atomic_cas_system,
    atomic_min,
    atomic_max,
    atomic_or,
    atomic_and,
)

# TMA (Tensor Memory Accelerator)
from .intrin.tma import (
    tma_load_2d,
    tma_load_multicast_2d,
    tma_store_2d,
    tma_store_fence,
    tma_store_arrive,
    tma_store_wait,
    tma_copy_1d_g2s,
    tma_copy_1d_s2g,
    tma_store_wait_n,
    prefetch_tma_descriptor,
)

# WGMMA (Warp Group MMA)
from .intrin.wgmma import (
    wgmma_fence,
    wgmma_commit,
    wgmma_wait,
    wgmma_m64n256k16,
    wgmma_m64n64k16,
    wgmma_init_accum,
    wgmma_zero_accum,
    wgmma_compute,
    wgmma_init_accum_64x64,
    wgmma_zero_accum_64x64,
    wgmma_compute_64x64,
    wgmma_fence_operand,
    wgmma_fence_operand_array,
    wgmma_fence_acc64,
    wgmma_fence_acc,
    store_acc64_to_global_f32,
    store_acc_f32_to_global,
    wgmma_init_4acc,
    wgmma_compute_00,
    wgmma_compute_01,
    wgmma_compute_10,
    wgmma_compute_11,
    wgmma_fence_4acc,
    store_4acc_to_global_f32,
    store_accum_swizzle,
    store_acc_to_global_n256,
    store_acc_to_smem_bf16_n256,
)

# UMMA / SM100 (Blackwell) intrinsics
from .intrin.umma import (
    elect_one,
    tmem_alloc,
    tmem_dealloc,
    tmem_load_4x,
    tmem_load_8x,
    tmem_load_fence,
    tcgen05_fence_after,
    tcgen05_fence_before,
    umma_f16_cg2,
    umma_commit_2sm,
    tma_load_2d_cg2,
    mbarrier_arrive_expect_tx_cluster,
    mbarrier_arrive_cluster,
    make_smem_desc_sm100,
    make_instr_desc,
    pack_bf16,
    st_shared_128,
    tma_store_2d_sm100,
    tma_store_commit,
    uint_as_float,
    tmem_store_bf16_row,
    tmem_epilogue_coalesced_4w,
)

# Loop utilities
from .intrin.loop import (unroll)

# Time utilities
from .intrin.time_utils import (clock64, globaltimer, nanosleep)

# Builtin base
from .builtin_base import builtin, Builtin
from .simple_builtin import simple_builtin, asm
